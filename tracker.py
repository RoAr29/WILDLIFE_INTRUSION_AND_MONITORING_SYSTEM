"""
tracker.py
----------
DeepSORT-based object tracker for the aerial surveillance drone system.

Connects to:
    <- detector.py          run_detection(frame) -> List[dict]
    -> alert logic          tracked objects + poacher_alert flag
    -> shared state         JSON / SQLite store

Input — each dict from run_detection():
    {
        "bbox":       [x, y, w, h],   # top-left-x, top-left-y, width, height
        "center":     [cx, cy],       # pre-computed by detector (ignored here)
        "confidence": float,
        "class_id":   int,
        "class_name": str,            # species name OR "human"
        "type":       str,            # "animal" | "human"
        "id":         None            # placeholder, filled here
    }

Output — dict returned by update():
    {
        "tracks": [
            {
                "track_id":   int | str,
                "bbox":       [x, y, w, h],
                "center":     [cx, cy],
                "class_name": str,
                "type":       str,        # "animal" | "human"
                "confidence": float
            },
            ...
        ],
        "poacher_alert": bool     # True when animals AND humans confirmed same frame
    }

BBOX FORMAT — critical:
    detector.py converts xyxy -> xywh internally via xyxy_to_xywh() before returning.
    deep_sort_realtime.update_tracks() natively expects [left, top, w, h]
    which is IDENTICAL to our [x, y, w, h].
    -> No conversion on input. No conversion on output.
    -> We never touch xyxy anywhere in this file.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort

from animal_model_classes import ANIMAL_CLASSES
from human_model_classes import HUMAN_CLASSES

# -- Build lookup sets from the class maps ---------------------------------
# Used to derive "type" independently of what detector.py sends,
# so even if detector logic changes, type resolution stays correct.

_ANIMAL_NAMES: set[str] = set(ANIMAL_CLASSES.values())   # {"sheep", "elephant", ...}
_HUMAN_NAMES:  set[str] = set(HUMAN_CLASSES.values())    # {"person"} -> normalised to "human"


def _resolve_type(class_name: str) -> str:
    """
    Derive canonical type from class_name using the ground-truth class maps.
    Returns "animal", "human", or "unknown".
    """
    if class_name in _ANIMAL_NAMES:
        return "animal"
    if class_name in _HUMAN_NAMES or class_name == "human":
        return "human"
    return "unknown"


# -- Tracker class ---------------------------------------------------------

class ObjectTracker:
    """
    Wraps DeepSORT to assign persistent track IDs across frames for both
    the animal model (10 species) and the human model (person class).

    Parameters
    ----------
    max_age : int
        Frames a track coasts without a matching detection before deletion.
        30 handles brief occlusions common in forest canopy drone footage.
    n_init : int
        Consecutive frames required to confirm a track.
        2 prevents single-frame ghost detections from reaching the alert engine.
    max_cosine_distance : float
        Re-ID appearance similarity threshold. 0.4 is a solid default for
        outdoor aerial footage where lighting changes frame to frame.
    nn_budget : int
        Max appearance-feature vectors stored per track gallery.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 2,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
    ):
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

        # class_name -> type cache.
        # Pre-populated from ground-truth class maps so coasting tracks
        # (get_det_conf() == None) always return correct type.
        self._class_to_type: dict[str, str] = {}
        for name in _ANIMAL_NAMES:
            self._class_to_type[name] = "animal"
        for name in _HUMAN_NAMES:
            self._class_to_type[name] = "human"
        self._class_to_type["human"] = "human"   # detector normalises "person" -> "human"

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def update(self, detections: list[dict], frame=None) -> dict:
        """
        Run DeepSORT on one frame's detections and return confirmed tracks.

        Parameters
        ----------
        detections : list[dict]
            Direct output of detector.run_detection(frame).
            Extra keys ("class_id", "id", "center") are safely ignored.
        frame : np.ndarray | None
            Current video frame in BGR (H, W, C).
            Required for DeepSORT appearance-embedding network.
            Pass None only in unit tests.

        Returns
        -------
        dict
            {
                "tracks":        list[dict],
                "poacher_alert": bool
            }
        """

        # -- Step 1: Build DeepSORT input ----------------------------------
        #
        # DeepSORT expects: List[ Tuple[ [left, top, w, h], confidence, class_name ] ]
        #
        # detector.py already converts:
        #   xyxy  --xyxy_to_xywh()--> [x, y, w, h]
        #
        # [x, y, w, h] == [left, top, w, h]  -> pass through directly.
        # NO xyxy conversion here. Ever.

        raw_detections = []
        for det in detections:
            class_name    = det["class_name"]
            resolved_type = _resolve_type(class_name)
            self._class_to_type[class_name] = resolved_type   # update cache

            raw_detections.append((
                det["bbox"],        # [x, y, w, h] -- correct format, no conversion
                det["confidence"],
                class_name,
            ))

        # -- Step 2: Run DeepSORT ------------------------------------------
        tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        # -- Step 3: Convert confirmed tracks -> output dicts -------------
        tracked_objects: list[dict] = []
        has_animal = False
        has_human  = False

        for track in tracks:
            # Tentative = seen < n_init consecutive frames -> skip
            if not track.is_confirmed():
                continue

            track_id   = track.track_id
            class_name = track.get_det_class()

            # get_det_conf() is None on coasting frames (no match this round)
            confidence = track.get_det_conf()
            if confidence is None:
                confidence = 0.0

            # to_ltwh(orig=True):
            #   matched frame  -> original detection bbox [x, y, w, h]
            #   coasting frame -> Kalman-predicted bbox   [x, y, w, h]
            # Both are already [x, y, w, h]. No post-processing needed.
            ltwh = track.to_ltwh(orig=True)
            if ltwh is None:
                ltwh = track.to_ltwh(orig=False)

            x, y, w, h = (float(v) for v in ltwh)
            cx = round(x + w / 2, 2)
            cy = round(y + h / 2, 2)

            det_type = self._class_to_type.get(class_name, "unknown")

            if det_type == "animal":
                has_animal = True
            elif det_type == "human":
                has_human = True

            tracked_objects.append({
                "track_id":   track_id,
                "bbox":       [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "center":     [cx, cy],
                "class_name": class_name,
                "type":       det_type,
                "confidence": round(float(confidence), 4),
            })

        # -- Step 4: Poacher alert ----------------------------------------
        #
        # Fired here on CONFIRMED tracks, not on raw detections like detector.py.
        # n_init=2 means a ghost bounding box appearing for just one frame
        # will never trigger a false poacher alarm.
        poacher_alert = has_animal and has_human

        return {
            "tracks":        tracked_objects,
            "poacher_alert": poacher_alert,
        }


# -- Singleton helper ------------------------------------------------------

_default_tracker: ObjectTracker | None = None


def get_tracker(**kwargs) -> ObjectTracker:
    """
    Return a module-level singleton ObjectTracker.
    kwargs only applied on the very first call.

    Usage in your pipeline:
        from tracker import get_tracker

        result = get_tracker().update(detections, frame=frame)
        tracks        = result["tracks"]
        poacher_alert = result["poacher_alert"]
    """
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ObjectTracker(**kwargs)
    return _default_tracker


# -- Smoke test  (python tracker.py) --------------------------------------

if __name__ == "__main__":
    import numpy as np

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Exact dict structure produced by detector.run_detection()
    # detector uses animal_model.names[cls] for animals -> "elephant" (class_id 7)
    # and hardcodes class_name = "human" for the person class
    fake_detections = [
        {
            "bbox":       [120, 80, 200, 150],   # [x, y, w, h]
            "center":     [220.0, 155.0],        # ignored by tracker
            "confidence": 0.87,
            "class_id":   7,                     # 7 = "elephant" per ANIMAL_CLASSES
            "class_name": "elephant",
            "type":       "animal",
            "id":         None,
        },
        {
            "bbox":       [500, 300, 60, 180],
            "center":     [530.0, 390.0],
            "confidence": 0.76,
            "class_id":   0,
            "class_name": "human",
            "type":       "human",
            "id":         None,
        },
    ]

    tracker = ObjectTracker()

    for frame_idx in range(4):
        result = tracker.update(fake_detections, frame=dummy_frame)
        print(f"\n-- Frame {frame_idx + 1} ------------------------------------------")
        print(f"  poacher_alert : {result['poacher_alert']}")
        if result["tracks"]:
            for obj in result["tracks"]:
                print(f"  {obj}")
        else:
            print("  (no confirmed tracks yet — waiting for n_init frames)")

    # -- Verify all 10 animal class names resolve correctly --
    print("\n-- Class map validation ------------------------------------")
    for cid, cname in ANIMAL_CLASSES.items():
        t = _resolve_type(cname)
        print(f"  class_id={cid:2d}  class_name={cname:<12s}  -> type={t}")
    print(f"  class_name={'human':<12s}               -> type={_resolve_type('human')}")

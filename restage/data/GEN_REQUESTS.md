# Requests to the generator (restage/gen) from the data pipeline

The data pipeline does not edit `restage/gen/`. Problems found while packing are listed here with a
reproduction; the pipeline works around each one as noted.

## 1. Non-finite camera positions (driving)

    python -m restage.gen --genre driving --n 1 --seconds 12 --res 448x252 --aux-res 256x144 --seed 3 --out /tmp/x

prints `cinema.py:584: RuntimeWarning: overflow encountered in cast` and `cam1` (shot s1, move
`truck`) contains `inf` positions (11 values). The visibility code then divides by inf
(`scenario.py:264`).

*Workaround:* `pack.py` rejects clips with non-finite or out-of-range camera tracks (`BadSample`),
and `make_dataset.py` moves on to the next seed of the split. The rejection rate is reported in the
dataset's `index.json` (`failures`).

## 2. Person appearance in the document (nice to have)

Entities carry no appearance metadata, so `pack.py` derives person colours (skin, top, bottom,
shoes, hair) from the vertex colours grouped by skinning joint and height. If the generator wrote
`entities[i].appearance = {"skin": [...], "top": [...], "bottom": [...], "shoes": [...], "hair": [...],
"height": h}` (linear RGB, metres) the packer would use it directly (it already looks for it).

## 3. One-frame camera jumps inside shots (all genres)

Camera tracks often turn by 5-22 degrees between two consecutive frames inside a shot, while the
surrounding frames move by < 0.5 degree, sometimes jumping back a few frames later. In the video this
looks exactly like a cut (the cut detector fires on it) but the edit has no cut there. Examples:

    python -m restage.gen --genre interview --n 1 --seconds 15.93 --seed 1000001 --errors --dr 0.461 \
        --res 448x252 --aux-res 320x180 --out /tmp/x --name iv
    # shot s1 (dolly): forward axis turns 8.3 deg at frames 112, 137 and 259 (and ~0.02 deg/frame elsewhere)
    python -m restage.gen --genre hiking --n 1 --seconds 15.34 --seed 2000003 --errors --dr 0.293 ...
    # pan / handheld shots: 20.4 deg at frame 25, 10.3 at 150, 20.6 at 245, 13.9 at 397, 424, 448
    python -m restage.gen --genre dance --n 1 --seconds 12 --seed 5 ...      # tilt shot: 10.9 deg at 294
    python -m restage.gen --genre driving --n 1 --seconds 12 --seed 5 ...    # truck shot: 13.5 deg at 213

Likely cause: `CameraSolver._screen_x` (cinema.py) returns a step function of the subject's facing
(`np.where(abs(look) > 0.25, -sign(look) * k, sign(look) * k * 0.3)`), so the horizontal framing target
switches between +-0.3 and -+0.09 in one frame whenever `look` crosses 0 or +-0.25. Smoothing x_n over time
(or making it a continuous function of `look`) should remove the jumps.

*Workaround:* `pack.camera_jumps` finds these frames (a turn whose excess over the local median is > 6% of the fov and
> 2.5x that median, or a move > 0.3 m and > 5x the local median) and stores them in `meta["cam_jumps"]`. The cut detector
ignores +-1 frame around them in training and evaluation (and reports F1 with them counted, too).
Keyframe cameras are still exact, so the model's labels are unaffected.

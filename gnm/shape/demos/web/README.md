# GNM Web Demo

[![GNM Head Explorer Preview](teaser/teaser.webm)](https://xrblocks.github.io/demos/gnm/)

**Live Demo:** [https://xrblocks.github.io/demos/gnm/](https://xrblocks.github.io/demos/gnm/)

This is a standalone web demonstration of the [GNM](https://github.com/google/gnm) (Generative aNthropometric Model) running purely in the Chrome browser
across desktop, mobile, and [Android XR](https://android.com/xr) devices.

The demo runs using [XR Blocks](https://github.com/google/xrblocks), loading it dynamically via CDN, which provides the 3D scene, rendering, and spatial UI. You can explore more interactive XR examples and the full SDK in the [XR Blocks repository](https://github.com/google/xrblocks).

## Running the Demo

Because this demo uses ES modules, you must serve it over HTTP (opening `index.html` directly from your file system via `file://` will not work).

You can use any local web server. For example, if you have Python installed, run this command in this directory:

```bash
# Python 3
python -m http.server 8000
```

Then, open your browser and navigate to:
[http://localhost:8000/](http://localhost:8000/)

## Model Assets

By default, the demo fetches the required model weights (`gnm_head_web.bin` and `gnm_samplers_web.bin`) dynamically from a CDN.

If you prefer to run the demo entirely offline, you can download the pre-converted `.bin` files from the [assets-gnm repository](https://github.com/xrblocks/assets-gnm/tree/main).

1. Create an `assets/` folder in this directory.
2. Place `gnm_head_web.bin` and `gnm_samplers_web.bin` inside `assets/`.
3. Open `main.js` and set `const USE_LOCAL_ASSETS = true;`, or append `?localAssets=1` to your URL.

Alternatively, you can generate the `.bin` files yourself from the main GNM repository using the script located in `tools/export_gnm_web.py`.

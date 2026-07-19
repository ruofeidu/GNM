import 'xrblocks/addons/simulator/SimulatorAddons.js';

import * as uikit from '@pmndrs/uikit';
import * as xb from 'xrblocks';

import {GNMControls} from './GNMControls.js';
import {GNMHeadModel} from './GNMModel.js';
import {GNMScene} from './GNMScene.js';
import {GNMSamplers} from './SemanticSampler.js';

// The GNM model weights are hosted on a CDN (pinned to a commit) rather than
// checked into the repo, since the head model alone is ~35 MB. They are the
// web export of gnm/shape/data (see tools/export_gnm_web.py).
const CDN_ASSETS_BASE =
  'https://rawcdn.githack.com/xrblocks/assets-gnm/8480138a42ae746a2f7c9808a51ef23af7648653';
// Local copies for offline development: generate them from a GNM checkout with
//   python tools/export_gnm_web.py --gnm_root=/path/to/gnm --out_dir=./assets
// then load them with the toggle below.
const LOCAL_ASSETS_BASE = './assets';

// DEBUG TOGGLE — flip to true to load the model from local ./assets instead of
// the public CDN. Can also be overridden per-load without editing code via the
// URL query: ?localAssets=1 (local) or ?localAssets=0 (CDN).
const USE_LOCAL_ASSETS = false;

const useLocalAssets = xb.getUrlParamBool('localAssets', USE_LOCAL_ASSETS);
const ASSETS_BASE = useLocalAssets ? LOCAL_ASSETS_BASE : CDN_ASSETS_BASE;
const HEAD_MODEL_URL = `${ASSETS_BASE}/gnm_head_web.bin`;
const SAMPLERS_URL = `${ASSETS_BASE}/gnm_samplers_web.bin`;
console.info(
  `[GNM] loading model assets from ${useLocalAssets ? 'LOCAL' : 'CDN'}: ${ASSETS_BASE}`
);

function setLoadingProgress(fraction, label) {
  const bar = document.querySelector('#gnm-loading .bar div');
  const text = document.querySelector('#gnm-loading p');
  if (bar) bar.style.width = `${Math.round(fraction * 100)}%`;
  if (label && text) text.textContent = label;
}

async function start() {
  try {
    let modelProgress = 0;
    let samplerProgress = 0;
    const report = () =>
      setLoadingProgress(
        modelProgress * 0.92 + samplerProgress * 0.08,
        'Downloading GNM model data…'
      );
    const [model, samplers] = await Promise.all([
      GNMHeadModel.load(HEAD_MODEL_URL, (p) => {
        modelProgress = p;
        report();
      }),
      GNMSamplers.load(SAMPLERS_URL, (p) => {
        samplerProgress = p;
        report();
      }),
    ]);
    setLoadingProgress(1, 'Starting XR Blocks…');

    const options = new xb.Options();
    options.enableUI();
    options.uikit.enable(uikit); // registers the uikit renderer for uiblocks
    options.enableReticles();
    options.setAppTitle('GNM Head Explorer');
    options.xrButton.startText = '<i id="xrlogo"></i> EXPLORE IN XR';
    options.xrButton.endText = '<i id="xrlogo"></i> EXIT XR';

    const scene = new GNMScene(model, samplers);
    xb.add(scene);
    window.gnm = {model, samplers, scene};

    const controls = new GNMControls(model, samplers, scene);
    controls.attach();
    document.getElementById('gnm-loading')?.remove();

    await xb.init(options);
  } catch (error) {
    setLoadingProgress(0, `Failed to load: ${error.message}`);
    console.error(error);
  }
}

document.addEventListener('DOMContentLoaded', start);

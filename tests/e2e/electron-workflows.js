// OpenDIAKiosk scenario for electron/scripts/e2e.js, run in the Electron
// window on the bundled example data:
// - "Targeted Data Extraction": loads the example DIA run with pyOpenMS and
//   extracts the target peptides; its 2D/3D plots must draw (3D needs WebGL).
// - "Statistical Validation": loads the example OpenSWATH features and runs
//   the semi-supervised scoring.
// - "Run Workflow": every executable the OpenSWATH pipeline calls (TOPP tools
//   and the bundled pyprophet) must be found on PATH.
const STEP_TIMEOUT = 15 * 60 * 1000;

async function plots(t, min, where) {
  const started = Date.now();
  let n = 0;
  while (Date.now() - started < 3 * 60 * 1000) {
    await t.assertNoException(where);
    const body = await t.page.locator("body").innerText();
    if (body.includes("WebGL is not supported")) throw new Error(`${where}: WebGL plots did not render`);
    n = await t.page.locator(".js-plotly-plot").count();
    if (n >= min) break;
    await t.page.waitForTimeout(2000);
  }
  t.log(`${where}: ${n} plots`);
  if (n < min) throw new Error(`${where}: expected at least ${min} plots, got ${n}`);
}

// Starts a step that computes during the rerun it triggers; t.clickButton
// would give up after the default two-minute settle, which the Windows
// runner exceeds for the semi-supervised scoring.
async function runStep(t, button, doneText) {
  await t.page.getByRole("button", { name: button }).first().click();
  await t.waitForText(doneText, { timeout: STEP_TIMEOUT });
  await t.settle({ timeout: STEP_TIMEOUT });
}

module.exports = async (t) => {
  await t.open("dia_01_targeted_data_extraction");
  await runStep(t, "▶ Load DIA Data", "Data loaded successfully.");
  await plots(t, 1, "Targeted Data Extraction (loaded)");
  await runStep(t, "▶ Perform Targeted Extraction", "Stage 3");
  await plots(t, 3, "Targeted Data Extraction (extracted)");
  await t.shot("targeted-extraction");

  await t.open("dia_04_statistical_validation");
  await t.clickButton("▶ Load Feature Data");
  await plots(t, 1, "Statistical Validation (loaded)");
  await runStep(t, "▶ Run Semi-Supervised Learning", "semi-supervised model performance");
  await t.shot("statistical-validation");

  await t.open("openswath_workflow");
  const body = await t.page.locator("body").innerText();
  for (const tool of ["OpenSwathAssayGenerator", "OpenSwathDecoyGenerator", "OpenSwathWorkflow", "pyprophet"]) {
    if (!body.includes(`${tool} found at`)) throw new Error(`Run Workflow: ${tool} is not found on PATH`);
  }
  await t.shot("run-workflow-checks");
};

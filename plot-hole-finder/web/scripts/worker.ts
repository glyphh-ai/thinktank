import { runWorker } from "../lib/worker";

runWorker().catch((err) => {
  console.error(err);
  process.exit(1);
});

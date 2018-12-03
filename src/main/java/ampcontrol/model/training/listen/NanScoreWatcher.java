package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

/**
 * Watches for non-finite (e.g. infinite, NaN) scores and notifies the provided callback {@link Runnable}.
 *
 * @author Christian Skärby
 */
public class NanScoreWatcher extends BaseTrainingListener {

    private final Runnable nanCallback;

    private final static class RunOnce implements Runnable {

        private Runnable runnable;

        private RunOnce(Runnable runnable) {
            this.runnable = runnable;
        }

        @Override
        public void run() {
            runnable.run();
            runnable = () -> {/* do nothing */};
        }
    }

    public NanScoreWatcher(Runnable nanCallback) {
        this.nanCallback = nanCallback;
    }

    /**
     * Returns a {@link NanScoreWatcher} which will only notify the listener once
     * @param nanCallback Callback
     * @return a {@link NanScoreWatcher}
     */
    public static NanScoreWatcher once(Runnable nanCallback) {
        return new NanScoreWatcher(new RunOnce(nanCallback));
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if(!Double.isFinite(model.score())) {
            nanCallback.run();
        }
    }
}

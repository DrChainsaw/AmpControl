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

    public NanScoreWatcher(Runnable nanCallback) {
        this.nanCallback = nanCallback;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if(!Double.isFinite(model.score())) {
            nanCallback.run();
        }
    }
}

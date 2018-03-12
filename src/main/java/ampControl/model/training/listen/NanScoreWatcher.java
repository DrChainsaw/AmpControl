package ampControl.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

/**
 * Watches for non-finite (e.g. infinite, NaN) scores and notifies the provided callback {@link Runnable}.
 *
 * @author Christian Skärby
 */
public class NanScoreWatcher implements IterationListener {

    private final Runnable nanCallback;

    public NanScoreWatcher(Runnable nanCallback) {
        this.nanCallback = nanCallback;
    }

    private boolean invoked = false;

    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
    invoked = true;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        invoke();
        if(!Double.isFinite(model.score())) {
            nanCallback.run();
        }
    }
}

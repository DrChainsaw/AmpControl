package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.function.Supplier;

/**
 * Supplies the last iteration number
 *
 * @author Christian Skärby
 */
public class IterationSupplier implements IterationListener, Supplier<Integer> {

    private int lastIter = 0;
    private boolean invoked = false;

    @Override
    public Integer get() {
        return lastIter;
    }

    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
        invoked = true;
    }

    @Override
    public void iterationDone(Model model, int i) {
        invoke();
        lastIter = i;
    }
}

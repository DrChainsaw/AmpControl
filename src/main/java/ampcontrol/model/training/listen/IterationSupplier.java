package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

import java.util.function.Supplier;

/**
 * Supplies the last iteration number
 *
 * @author Christian Skärby
 */
public class IterationSupplier extends BaseTrainingListener implements Supplier<Integer> {

    private int lastIter = 0;

    @Override
    public Integer get() {
        return lastIter;
    }

    @Override
    public void iterationDone(Model model, int i, int epoch) {
        lastIter = i;
    }
}

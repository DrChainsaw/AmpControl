package ampControl.amp.probabilities;

import java.util.Collections;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * {@link Interpreter} which maps the given {@link INDArray} to its maximum index.
 *
 * @author Christian Skärby
 */
public class ArgMax implements Interpreter<Integer> {

    @Override
    public List<Integer> apply(INDArray probabilities) {
        return Collections.singletonList(probabilities.argMax(1).getInt(0));
    }
}

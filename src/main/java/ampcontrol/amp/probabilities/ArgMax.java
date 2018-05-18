package ampcontrol.amp.probabilities;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;

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

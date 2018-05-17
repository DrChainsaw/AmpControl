package ampcontrol.amp.probabilities;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Generic {@link Interpreter} which masks classifications for which the probability is below a given threshold.
 *
 * @param <T>
 *
 * @author Christian Skärby
 */
public class ThresholdFilter<T> implements Interpreter<T> {

    private final Interpreter<T> next;
    private final int index;
    private final double threshold;

    public ThresholdFilter(int index, double threshold, Interpreter<T> next) {
        this.index = index;
        this.threshold = threshold;
        this.next = next;
    }

    @Override
    public List<T> apply(INDArray indArray) {
        if(indArray.argMax(1).getInt(0) == index) {
            if (indArray.getDouble(index) < threshold) {
                return new ArrayList<>();
            }
        }
        return next.apply(indArray);
    }
}

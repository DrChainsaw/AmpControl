package ampControl.audio.processing;

import java.util.Collections;
import java.util.List;

/**
 * Input {@link ProcessingResult} created from a single double array
 *
 * @author Christian Skärby
 */
public class SingletonDoubleInput implements ProcessingResult {

    private final double[][] data;

    public SingletonDoubleInput(double[] inputVec) {
        this(new double[][] {inputVec});
    }

    public SingletonDoubleInput(double[][] data) {
        this.data = data;
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(data);
    }
}

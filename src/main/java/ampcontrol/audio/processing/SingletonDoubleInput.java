package ampcontrol.audio.processing;

import java.util.stream.Stream;

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
    public Stream<double[][]> stream() {
        return Stream.<double[][]>builder().add(data).build();
    }
}

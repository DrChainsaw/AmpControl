package ampControl.audio.processing;

import java.util.Collections;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;

public class TestProcessing implements ProcessingResult.Processing {

    private final DoubleUnaryOperator resultFunction;
    private final String name;
    private double[][] result;

    public TestProcessing(DoubleUnaryOperator resultFunction, String name) {
        this.resultFunction = resultFunction;
        this.name = name;
    }

    @Override
    public void receive(double[][] input) {
        result = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            result[i] = DoubleStream.of(input[i]).map(resultFunction).toArray();
        }
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(result);
    }
}

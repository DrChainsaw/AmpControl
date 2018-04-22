package ampControl.audio.processing;

import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public class TestProcessing implements ProcessingResult.Factory {

    private final DoubleUnaryOperator resultFunction;
    private final String name;

    public TestProcessing(DoubleUnaryOperator resultFunction, String name) {
        this.resultFunction = resultFunction;
        this.name = name;
    }
    @Override
    public ProcessingResult create(ProcessingResult input) {
        return new Result(input);
    }

    private final class Result implements ProcessingResult {

        private final ProcessingResult input;

        public Result(ProcessingResult input) {
            this.input = input;
        }

        @Override
        public List<double[][]> get() {
            return input.get().stream().map(inputArr -> {
                final double[][] result = new double[inputArr.length][];
                for (int i = 0; i < inputArr.length; i++) {
                    result[i] = DoubleStream.of(inputArr[i]).map(resultFunction).toArray();
                }
                return result;

            }).collect(Collectors.toList());
        }
    }

    @Override
    public String name() {
        return name;
    }
}

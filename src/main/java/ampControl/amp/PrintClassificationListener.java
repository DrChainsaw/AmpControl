package ampControl.amp;

import com.beust.jcommander.Parameter;
import ampControl.admin.param.IntToDoubleConverter;
import ampControl.amp.probabilities.ArgMax;
import ampControl.amp.probabilities.Interpreter;
import ampControl.amp.probabilities.ThresholdFilter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * {@link ClassificationListener} which prints the result of classification in the terminal.
 *
 */
public class PrintClassificationListener implements  ClassificationListener {

    public static final String RESET = "\033[0m";      // RESET as color sticks otherwise
    public static final String RED_BOLD = "\033[1;31m";    // RED
    public static final String GREEN_BOLD = "\033[1;32m";  // GREEN
    public static final String PURPLE_BOLD = "\033[1;35m"; // PURPLE
    public static final String BLUE_BOLD = "\033[1;34m";   // BLUE


    public static class Factory implements ClassificationListener.Factory {

        @Parameter(names = {"-labelToThreshold", "-ltt"},
                description = "Comma separated list of how to map labels programs. Syntax is <labelx>:<thredholsx>,<labely>:<thresholdy,...>",
                converter = IntToDoubleConverter.class)
        private Map<Integer, Double> probabilityThresholds = new IntToDoubleConverter().convert("" +
                "0:0.8," +
                "1:0.8," +
                "2:0.85," +
                "3:0.95");

        @Override
        public PrintClassificationListener create() {
            Interpreter<Integer> interpretProbabilities = new ArgMax();
            for (Map.Entry<Integer, Double> labelThreshEntry : probabilityThresholds.entrySet()) {
                interpretProbabilities = new ThresholdFilter<>(labelThreshEntry.getKey(), labelThreshEntry.getValue(), interpretProbabilities);
            }
            return new PrintClassificationListener(interpretProbabilities);
        }
    }

    private final Interpreter<Integer> interpreter;
    private final List<String> labelMap;
    private Consumer<String> labelConsumer = str -> System.out.println(str);


    public PrintClassificationListener(Interpreter<Integer> interpreter) {
        this.interpreter = interpreter;
        labelMap = Stream.of(
                GREEN_BOLD + "SILENCE" + RESET,
                RED_BOLD + "NOISE" + RESET,
                BLUE_BOLD + "RYTHM" + RESET,
                PURPLE_BOLD + "LEAD" + RESET).collect(Collectors.toList());
    }

    @Override
    public void indicateAudioClassification(INDArray probabilities) {
        List<Integer> labs = interpreter.apply(probabilities);
        if(labs.size() > 0) {
            labelConsumer.accept(labelMap.get(labs.get(0)));
        }
    }

    /**
     * Sets the label consumer. Mainly intended for testing.
     *
     * @param labelConsumer
     */
    void setLabelConsumer(Consumer<String> labelConsumer) {
        this.labelConsumer = labelConsumer;
    }
}

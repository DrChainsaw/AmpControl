package ampcontrol.amp;

import ampcontrol.admin.param.IntToDoubleConverter;
import ampcontrol.admin.service.Service;
import ampcontrol.admin.service.control.SubscriptionRegistry;
import ampcontrol.amp.midi.MidiChannelPar;
import ampcontrol.amp.midi.MidiServiceFactory;
import ampcontrol.amp.probabilities.ArgMax;
import ampcontrol.amp.probabilities.Interpreter;
import ampcontrol.amp.probabilities.ThresholdFilter;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.sound.midi.ShortMessage;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * {@link ClassificationListener} which prints the result of classification in the terminal.
 */
public class PrintClassificationListener implements AmpInterface {

    private static final String RESET = "\033[0m";      // RESET as color sticks otherwise
    private static final String RED_BOLD = "\033[1;31m";    // RED
    private static final String GREEN_BOLD = "\033[1;32m";  // GREEN
    private static final String PURPLE_BOLD = "\033[1;35m"; // PURPLE
    private static final String BLUE_BOLD = "\033[1;34m";   // BLUE

    private final Interpreter<Integer> interpreter;
    private final Service serviceDelegate;
    private final List<String> labelMap;
    private Consumer<String> labelConsumer = System.out::println;

    public static class Factory implements AmpInterface.Factory {

        @Parameter(names = {"-labelToThreshold", "-ltt"},
                description = "Comma separated list of how to map labels programs. Syntax is <labelx>:<thredholsx>,<labely>:<thresholdy,...>",
                converter = IntToDoubleConverter.class)
        private Map<Integer, Double> probabilityThresholds = new IntToDoubleConverter().convert("" +
                "0:0.8," +
                "1:0.8," +
                "2:0.85," +
                "3:0.95");

        @ParametersDelegate
        private final MidiChannelPar midiChannelPar = new MidiChannelPar();

        @ParametersDelegate
        private final MidiServiceFactory midiServiceFactory = new MidiServiceFactory(midiChannelPar);

        @Override
        public PrintClassificationListener create() {
            Interpreter<Integer> interpretProbabilities = new ArgMax();
            for (Map.Entry<Integer, Double> labelThreshEntry : probabilityThresholds.entrySet()) {
                interpretProbabilities = new ThresholdFilter<>(labelThreshEntry.getKey(), labelThreshEntry.getValue(), interpretProbabilities);
            }
            return new PrintClassificationListener(interpretProbabilities, midiServiceFactory.createService(msg -> System.out.println(msgToString(msg))));
        }
    }

    private PrintClassificationListener(Interpreter<Integer> interpreter, Service serviceDelegate) {
        this.interpreter = interpreter;
        this.serviceDelegate = serviceDelegate;
        labelMap = Stream.of(
                GREEN_BOLD + "SILENCE" + RESET,
                RED_BOLD + "NOISE" + RESET,
                BLUE_BOLD + "RYTHM" + RESET,
                PURPLE_BOLD + "LEAD" + RESET).collect(Collectors.toList());
    }

    @Override
    public void indicateAudioClassification(INDArray probabilities) {
        List<Integer> labs = interpreter.apply(probabilities);
        if (labs.size() > 0) {
            labelConsumer.accept(labelMap.get(labs.get(0)));
        }
    }

    @Override
    public void registerTo(SubscriptionRegistry subscriptionRegistry) {
        serviceDelegate.registerTo(subscriptionRegistry);
    }

    private static String msgToString(ShortMessage msg) {
        return new StringBuilder()
                .append("Got Midi message! Command: ")
                .append(msg.getCommand())
                .append(", channel: ")
                .append(msg.getChannel())
                .append(", data1: ")
                .append(msg.getData1())
                .append(", data2: ")
                .append(msg.getData2())
                .toString();
    }

    /**
     * Sets the label consumer. Mainly intended for testing.
     *
     * @param labelConsumer consumes labels
     */
    void setLabelConsumer(Consumer<String> labelConsumer) {
        this.labelConsumer = labelConsumer;
    }
}

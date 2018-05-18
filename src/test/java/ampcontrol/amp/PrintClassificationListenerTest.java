package ampcontrol.amp;

import com.beust.jcommander.JCommander;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class PrintClassificationListenerTest {

    private final static String labelToThreshPar = "-ltt ";

    @Test
    public void indicateAudioClassification() {
        final List<Double> pThresholds = Arrays.asList(0.5, 0.7, 0.8, 0.3);
        final String argStr =
                labelToThreshPar +
                        IntStream.range(0, pThresholds.size()).mapToObj(lab -> lab + ":" + pThresholds.get(lab))
                                .collect(Collectors.joining(","));

        final PrintClassificationListener.Factory factory = new PrintClassificationListener.Factory();
        JCommander.newBuilder().addObject(factory)
                .build()
                .parse(argStr.split(" "));

        final PrintClassificationListener printListener = factory.create();

        printListener.setLabelConsumer(str -> fail("Did not expect output " + str + "!"));
        printListener.indicateAudioClassification(Nd4j.create(new double[]{0.49, 0.69, 0.79, 0.29}));

        callAndAssert(printListener, "SILENCE", new double[]{0.51, 0.0, 0.0, 0.49});
        callAndAssert(printListener, "NOISE",   new double[]{0.2, 0.71, 0.09, 0.0});
        callAndAssert(printListener, "RYTHM",   new double[]{0.1, 0.08, 0.82, 0.0});
        callAndAssert(printListener, "LEAD",    new double[]{0.2, 0.2, 0.29, 0.31});

    }

    void callAndAssert(PrintClassificationListener printListener, String expected, double[] probVec) {
        final LabelProbe probe = new LabelProbe(expected);
        final INDArray probs = Nd4j.create(probVec);
        printListener.setLabelConsumer(probe);
        printListener.indicateAudioClassification(probs);
        probe.assertWasCalled();
    }

    private final class LabelProbe implements Consumer<String> {

        private final String expected;
        private boolean wasCalled = false;

        public LabelProbe(String expected) {
            this.expected = expected;
        }

        @Override
        public void accept(String str) {
            wasCalled = true;
            assertTrue("Incorrect label!", str.contains(expected));
        }

        void assertWasCalled() {
            assertTrue("Was not called when expecting " + expected + "!", wasCalled);
        }
    }
}
package ampcontrol.admin.service.classification;

import ampcontrol.admin.service.MockControlRegistry;
import ampcontrol.admin.service.classifiction.AudioClassificationService;
import ampcontrol.amp.ClassificationListener;
import ampcontrol.model.inference.Classifier;
import com.beust.jcommander.JCommander;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

import static org.junit.Assert.*;

/**
 * Test cases for {@link AudioClassificationService}
 *
 * @author Christian Skärby
 */
public class AudioClassificationServiceTest {

    private final static int maxNrofRetries = 100;

    private final static String minTimeBetweenUpdatesPar = "-minTimeBetweenUpdates ";
    private final static String actMsgPar = "-actAutoMsg ";

    /**
     * Test that service can be started and stopped and resumed correctly
     */
    @Test
    public void startStop() {
        final AudioClassificationService service = new AudioClassificationService();
        final int msBetweenUpdates = 2;
        final String actMsg = "gjkflghjghht";
        final String params = minTimeBetweenUpdatesPar + msBetweenUpdates + " " + actMsgPar + actMsg;

        JCommander.newBuilder().addObject(service)
                .build()
                .parse(params.split(" "));

        final MockControlRegistry registry = new MockControlRegistry();

        final ProbeClassificationListener listenerProbe = new ProbeClassificationListener();
        service.initialize(listenerProbe, dummyClassifier, () -> {/* Do nothing*/});
        service.registerTo(registry);

        assertFalse("Service runs without being started!", service.isRunning());

        registry.execute(actMsg);

        assertTrue("Service not running!", service.isRunning());

        waitForCondition(() -> listenerProbe.nrofCalls > 0,  msBetweenUpdates);
        service.stop();
        assertFalse("Service runs after stopped!", service.isRunning());

        listenerProbe.assertNrofCallsMoreThan(0);
        final int lastNrofCalls = listenerProbe.nrofCalls;

        registry.execute(actMsg);
        assertTrue("Service not running!", service.isRunning());

        waitForCondition( () -> listenerProbe.nrofCalls > lastNrofCalls,  msBetweenUpdates);
        service.stop();
        assertFalse("Service runs after stopped!", service.isRunning());

        listenerProbe.assertNrofCallsMoreThan(lastNrofCalls);

    }

    private static void waitForCondition(Supplier<Boolean> condition, long sleepTimeMs) {
        try {
            int retryCnt = 0;
            while (!condition.get() && retryCnt < maxNrofRetries) {
                Thread.sleep(sleepTimeMs);
                retryCnt++;
            }
        } catch (InterruptedException e) {
            fail("test interrupted!");
        }
    }

    private static class ProbeClassificationListener implements ClassificationListener {

        private int nrofCalls = 0;

        @Override
        public void indicateAudioClassification(INDArray probabilities) {
            nrofCalls++;
        }

        void assertNrofCallsMoreThan(int expected) {
            assertTrue("Incorrect number of calls! Expected more than " + expected + ", was " + nrofCalls,
                    nrofCalls > expected);
        }
    }

    private final static Classifier dummyClassifier = new Classifier() {
        @Override
        public INDArray classify() {
            return null;
        }

        @Override
        public double getAccuracy() {
            return 0;
        }
    };
}
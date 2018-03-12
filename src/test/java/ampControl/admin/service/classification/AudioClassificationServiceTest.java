package ampControl.admin.service.classification;

import com.beust.jcommander.JCommander;
import ampControl.admin.service.classifiction.AudioClassificationService;
import ampControl.amp.ClassificationListener;
import ampControl.model.inference.Classifier;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.Assert.*;

/**
 * Test cases for {@link AudioClassificationService}
 *
 * @author Christian Skärby
 *
 */
public class AudioClassificationServiceTest {


    /**
     * Test that service can be started and stopped and resumed correctly
     */
    @Test
    public void startStop() {
        final AudioClassificationService service = new AudioClassificationService();
        final int msBetweenUpdates = 2;
        final int nrofCyclesToRun = 2;
        final String params = "-minTimeBetweenUpdates " + msBetweenUpdates;

        JCommander.newBuilder().addObject(service)
                .build()
                .parse(params.split(" "));

        final ProbeClassificationListener listenerProbe = new ProbeClassificationListener();
        service.initialize(listenerProbe, dummyClassifier, () -> {/* Do nothing*/});

        assertFalse("Service runs without being started!", service.isRunning());

        service.start();
        assertTrue("Service not running!", service.isRunning());

        try {
            Thread.sleep(nrofCyclesToRun*msBetweenUpdates);
            service.stop();
            assertFalse("Service runs after stopped!", service.isRunning());

            listenerProbe.assertNrofCallsMoreThan(0);
            final int lastNrofCalls = listenerProbe.nrofCalls;

            service.start();
            assertTrue("Service not running!", service.isRunning());

            Thread.sleep(nrofCyclesToRun*msBetweenUpdates);
            service.stop();
            assertFalse("Service runs after stopped!", service.isRunning());

            listenerProbe.assertNrofCallsMoreThan(lastNrofCalls);

        } catch (InterruptedException e) {
            fail("Testing interruped!");
        }

    }

    private static class ProbeClassificationListener implements ClassificationListener {

        private int nrofCalls = 0;

        @Override
        public void indicateAudioClassification(INDArray probabilities) {
            nrofCalls++;
        }

        void assertNrofCallsMoreThan(int expected) {
            assertTrue("Incorrect number of calls! Expected more than " + expected +", was " + nrofCalls,
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
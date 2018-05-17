package ampcontrol.amp;

import ampcontrol.admin.service.control.TopicPublisher;
import com.beust.jcommander.JCommander;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class PublishingClassificationListenerTest {

    private final static String topicPar = "-mqttClassificationTopic ";
    private final static String probThreshPar = "-pth ";
    private final static String momThreshPar = "-mth ";
    private final static String prohibPar = "-upt ";

    @Test
    public void noClassification() {
        final String topic = "QREkjhntlrn";
        final String argStr =
                topicPar + topic;

        final TopicPublisher probe = (top, msg) -> fail("did not expect to be called!");
        final ClassificationListener pubListener = createClassificationListener(argStr, probe);
        pubListener.indicateAudioClassification(Nd4j.create(new double[]{0.1, 0.0, 0.0, 0.49}));
    }

    @Test
    public void indicateSilenceClassification() {
        final String topic = "hgjkhgfrbtrn";
        final double probThresh = 0.8;
        final int momThresh = 2;
        final int prohib = 0;
        final String argStr =
                topicPar + topic +
                        " " + probThreshPar + probThresh +
                        " " + momThreshPar + momThresh +
                        " " + prohibPar + prohib;

        final int nrofLabels = 10;
        try {
            for (int i = 0; i < nrofLabels; i++) {
                final LabelProbe probe = new LabelProbe(topic, String.valueOf(i));
                final ClassificationListener pubListener = createClassificationListener(argStr, probe);
                final INDArray probs = Nd4j.zeros(nrofLabels);
                probs.putScalar(i, probThresh * 1.001);
                Thread.sleep(2);
                pubListener.indicateAudioClassification(probs);
                Thread.sleep(2);
                pubListener.indicateAudioClassification(probs);
                Thread.sleep(2);
                pubListener.indicateAudioClassification(probs);

                probe.assertWasCalled();
            }
        } catch (InterruptedException e) {
            fail("Test aborted!");
        }
    }

    private static ClassificationListener createClassificationListener(String argStr, TopicPublisher probe) {
        final PublishingClassificationListener.Factory factory = new PublishingClassificationListener.Factory(probe);
        JCommander.newBuilder().addObject(factory)
                .build()
                .parse(argStr.split(" "));

        return factory.create();
    }

    private final class LabelProbe implements TopicPublisher {

        private final String expectedTopic;
        private final String expectedMsg;
        private boolean wasCalled = false;

        public LabelProbe(String expectedTopic, String expectedMsg) {
            this.expectedTopic = expectedTopic;
            this.expectedMsg = expectedMsg;
        }

        @Override
        public void publish(String topic, String message) {
            wasCalled = true;
            assertEquals("Incorrect topic", expectedTopic, topic);
            assertEquals("Incorrect label!", expectedMsg, message);
        }

        void assertWasCalled() {
            assertTrue("Was not called when expecting " + expectedMsg + "!", wasCalled);
        }
    }
}
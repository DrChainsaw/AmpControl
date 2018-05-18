package ampcontrol.model.training.data.iterators.preprocs;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link Cnn2DtoCnn1DInputPreprocessor}
 *
 * @author Christian Skärby
 */
public class Cnn2DtoCnn1DInputPreprocessorTest {

    /**
     * Test preProcess
     */
    @Test
    public void preProcess() {
        final int[] featureShapeIn = {3,1,7,11};
        final int[] expectedFeatureShape = {3,7,11};
        final InputPreProcessor pp = new Cnn2DtoCnn1DInputPreprocessor();
        assertArrayEquals("Incorrect shape!", expectedFeatureShape,
                pp.preProcess(Nd4j.create(featureShapeIn),1).shape());
    }

    /**
     * Test preProcess with too many channels
     */
    @Test(expected = RuntimeException.class)
    public void cnnToRnnFeatureTooManyChannels() {
        final int[] featureShapeIn = {3,2,7,11};
        new Cnn2DtoCnn1DInputPreprocessor().preProcess(Nd4j.create(featureShapeIn),1);
    }

}
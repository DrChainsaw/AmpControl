package ampcontrol.model.training.data.iterators.preprocs;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link CnnToManyToOneRnnPreProcessor}.
 *
 * @author Christian Skärby
 */
public class CnnToManyToOneRnnPreProcessorTest {

    /**
     * Test preProcess
     */
    @Test
    public void preProcess() {
        final long[] featureShapeIn = {3,1,7,11};
        final long[] expectedFeatureShape = {3,11,7};
        final long[] labelsShapeIn = {3,5};
        final long[] expectedLabelsShape = {3,5,7};
        final double[] expectedMask = {0,0,0,0,0,0,1};
        final INDArray expectedLabelsMask = Nd4j.create(new double[][] {expectedMask, expectedMask, expectedMask});
        final DataSet testSet = new org.nd4j.linalg.dataset.DataSet(Nd4j.create(featureShapeIn), Nd4j.create(labelsShapeIn));
        final DataSetPreProcessor pp = new CnnToManyToOneRnnPreProcessor();
        pp.preProcess(testSet);
        assertArrayEquals("Incorrect feature shape!", expectedFeatureShape, testSet.getFeatures().shape());
        assertArrayEquals("Incorrect labels shape!", expectedLabelsShape, testSet.getLabels().shape());
        assertArrayEquals("Incorrect mask!", expectedLabelsMask.reshape(expectedLabelsMask.length()).toDoubleVector(),
                testSet.getLabelsMaskArray().reshape(testSet.getLabelsMaskArray().length()).toDoubleVector(), 1e-10);

    }

    /**
     * Test cnnToRnnFeature
     */
    @Test
    public void cnnToRnnFeature() {
        final long[] featureShapeIn = {3,1,7,11};
        final long[] expectedFeatureShape = {3,11,7};
        assertArrayEquals("Incorrect shape!", expectedFeatureShape,
                CnnToManyToOneRnnPreProcessor.cnnToRnnFeature(Nd4j.create(featureShapeIn)).shape());
    }


    /**
     * Test cnnToRnnFeature with too many channels
     */
    @Test(expected = RuntimeException.class)
    public void cnnToRnnFeatureTooManyChannels() {
        final int[] featureShapeIn = {3,2,7,11};
        CnnToManyToOneRnnPreProcessor.cnnToRnnFeature(Nd4j.create(featureShapeIn)).shape();
    }
}
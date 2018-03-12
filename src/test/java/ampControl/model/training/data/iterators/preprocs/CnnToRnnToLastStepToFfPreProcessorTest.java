package ampControl.model.training.data.iterators.preprocs;

import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * Test cases for {@link CnnToRnnToLastStepToFfPreProcessor}.
 *
 * @author Christian Skärby
 */
public class CnnToRnnToLastStepToFfPreProcessorTest {

    /**
     * Test preProcess
     */
    @Test
    public void preProcess() {
        final int[] featureShapeIn = {3,1,7,11};
        final int[] expectedFeatureShape = {3,11,7};
        final int[] labelsShapeIn = {3,5};
        final int[] expectedLabelsShape = {3,5};
        final DataSet testSet = new org.nd4j.linalg.dataset.DataSet(Nd4j.create(featureShapeIn), Nd4j.create(labelsShapeIn));
        final DataSetPreProcessor pp = new CnnToRnnToLastStepToFfPreProcessor();
        pp.preProcess(testSet);
        assertArrayEquals("Incorrect feature shape!", expectedFeatureShape, testSet.getFeatures().shape());
        assertArrayEquals("Incorrect labels shape!", expectedLabelsShape, testSet.getLabels().shape());
    }
}
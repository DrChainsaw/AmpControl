package ampcontrol.model.training.model.evolve.transfer;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link MergeTransferTask}
 *
 * @author Christian Skärby
 */
public class MergeTransferTaskTest {

    /**
     * Test to transfer two 1D vectors out of which the first has changed size
     */
    @Test
    public void transferFirstSource1D() {
        final long length1 = 5;
        final long length2 = 8;
        final INDArray source1 = Nd4j.linspace(0, length1 - 1, length1);
        final INDArray target1 = Nd4j.zeros(1,source1.length() - 2);
        final INDArray source2 = Nd4j.linspace(source1.length(), source1.length() + length2 - 1, length2);
        final INDArray target2 = Nd4j.zeros(1,source2.length());

        final INDArray mergedSource = Nd4j.linspace(0, length1 + length2 - 1, length1 + length2);
        final INDArray mergedTarget = Nd4j.zeros(1,length1 + length2 - 2);


        final int[] expectedIndexes = {0,2,4,1,3};
        final INDArray expectedTarget1 = source1.get(new SpecifiedIndex(expectedIndexes)).get(NDArrayIndex.interval(0, target1.length())).transpose();
        final INDArray expectedTarget2 = source2;
        final INDArray expectedMergedTarget = Nd4j.concat(1,expectedTarget1, expectedTarget2);

        final TransferRegistry registry = new TransferRegistry();
        MergeTransferTask.builder()
                .addInput(source1.shape(), target1.shape(),
                        SingleTransferTask.builder()
                                .compFactory(SingleTransferTaskTest.fixedOrderComp(expectedIndexes))
                                .source(SingleTransferTask.IndMapping.builder()
                                        .entry(registry.register(source1, "source1"))
                                        .build())
                                .target(SingleTransferTask.IndMapping.builder()
                                        .entry(registry.register(target1, "target1"))
                                        .build()))
                .addInput(source2.shape(), target2.shape(),
                        SingleTransferTask.builder()
                                .source(SingleTransferTask.IndMapping.builder()
                                        .entry(registry.register(source2, "source1"))
                                        .build())
                                .target(SingleTransferTask.IndMapping.builder()
                                        .entry(registry.register(target2, "target1"))
                                        .build()))
                .addDependentTask(new SingleTransferTask.Builder()
                        .source(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(mergedSource, "mergeSource")).build())
                        .target(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(mergedTarget, "mergedTarget")).build()))
                .build().execute();

        registry.commit();

        assertEquals("Incorrect target1!", expectedTarget1, target1);
        assertEquals("Incorrect target2!", expectedTarget2, target2);
        assertEquals("Incorrect mergedTarget!", expectedMergedTarget, mergedTarget);


    }
}
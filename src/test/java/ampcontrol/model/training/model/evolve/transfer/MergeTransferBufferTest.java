package ampcontrol.model.training.model.evolve.transfer;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link MergeTransferBuffer}
 *
 * @author Christian Skärby
 */
public class MergeTransferBufferTest {

    /**
     * Test to transfer two 1D vectors out of which the first has changed size
     */
    @Test
    public void transferFirstSource1D() {
        final long length1 = 5;
        final long length2 = 8;
        final INDArray source1 = Nd4j.linspace(0, length1 - 1, length1);
        final INDArray target1 = Nd4j.zeros(1, source1.length() - 2);
        final INDArray source2 = Nd4j.linspace(source1.length(), source1.length() + length2 - 1, length2);
        final INDArray target2 = Nd4j.zeros(1, source2.length());

        final INDArray mergedSource = Nd4j.linspace(0, length1 + length2 - 1, length1 + length2);
        final INDArray mergedTarget = Nd4j.zeros(1, length1 + length2 - 2);


        final int[] expectedIndexes = {0, 2, 4, 1, 3};
        final INDArray expectedTarget1 = source1.get(new SpecifiedIndex(expectedIndexes)).get(NDArrayIndex.interval(0, target1.length())).transpose();
        final INDArray expectedMergedTarget = Nd4j.concat(1, expectedTarget1, source2);

        final TransferRegistry registry = new TransferRegistry();
        final TransferTask.ListBuilder builder1 = getListBuilder(source1, target1, registry)
                .compFactory(SingleTransferTaskTest.fixedOrderComp(expectedIndexes));
        final TransferTask.ListBuilder builder2 = getListBuilder(source2, target2, registry);
        final TransferTask.ListBuilder mergeBuilder = getListBuilder(mergedSource, mergedTarget, registry);

        MergeTransferBuffer.Builder mergeTransferTask = MergeTransferBuffer.builder()
                .addInput(source1.shape(), target1.shape(), builder1)
                .addInput(source2.shape(), target2.shape(), builder2)
                .addDependentTask(mergeBuilder);

        builder1.build().execute();
        builder2.build().execute();
        mergeTransferTask.build().transferBufferedIndexes();

        registry.commit();

        assertEquals("Incorrect target1!", expectedTarget1, target1);
        assertEquals("Incorrect target2!", source2, target2);
        assertEquals("Incorrect mergedTarget!", expectedMergedTarget, mergedTarget);
    }

    private SingleTransferTask.Builder getListBuilder(
            INDArray source,
            INDArray target,
            TransferRegistry registry) {
        return SingleTransferTask.builder()
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(source))
                        .build())
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(target))
                        .build());
    }

}
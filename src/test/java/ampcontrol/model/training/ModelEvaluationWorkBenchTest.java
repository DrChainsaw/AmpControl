package ampcontrol.model.training;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.model.evolve.GraphUtils;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link ModelEvaluationWorkBench}
 *
 * @author Christian Skärby
 */
public class ModelEvaluationWorkBenchTest {

    /**
     * Smoke test as nothing is really produced except logs
     */
    @Test
    public void evalute() {
        final DataSet dummySet = new DataSet(Nd4j.randn(new long[] {1,3,9,9}), Nd4j.randn(new long[] {1,2}));
        new ModelEvaluationWorkBench(
                new CachingDataSetIterator(new TestDataSetIterator(dummySet)),
                new CachingDataSetIterator(new TestDataSetIterator(dummySet)))
                .evalute(GraphUtils.getConvToDenseGraph("l1", "l2", "l3"), "l1");
    }
}
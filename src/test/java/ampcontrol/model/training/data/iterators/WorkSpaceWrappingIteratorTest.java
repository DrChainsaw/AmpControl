package ampcontrol.model.training.data.iterators;

import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Test cases for {@link WorkSpaceWrappingIterator}
 *
 * @author Christian Skärby
 */
public class WorkSpaceWrappingIteratorTest extends DecoratingMiniEpochIteratorTest {

    @Override
    protected MiniEpochDataSetIterator decorateMiniEpochIter(MiniEpochDataSetIterator anIterator) {
        return new WorkSpaceWrappingIterator(anIterator);
    }

    @Test
    public void testWsWrap() {
        final MiniEpochDataSetIterator sourceIter = new MockMiniEpochDataSetIterator() {
            @Override
            public DataSet next() {
                return new DataSet(Nd4j.create(1), Nd4j.create(1));
            }
        };
        assertFalse("Source shall not give attached data!", sourceIter.next().getFeatures().isAttached());

        final MiniEpochDataSetIterator testIter = new WorkSpaceWrappingIterator(sourceIter);

        assertTrue("Data shall be attched to workspace!", testIter.next().getFeatures().isAttached());
    }


}
package ampcontrol.model.training.data.iterators;

import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link DataSetCache}
 *
 * @author Christian Skärby
 */
public class DataSetCacheTest {

    /**
     * Test the check method
     */
    @Test
    public void cacheLoaded() {
        final DataSetCache cache = new DataSetCache();
        assertFalse("Does not expect to be loaded", cache.isLoaded());
        assertFalse("Does not expect to have next", cache.hasNext());

        final List<DataSet> expectedDs = Stream.generate(DataSet::new).limit(5).collect(Collectors.toList());
        cache.check(new MockDataSetIterator() {

            int cnt = -1;

            @Override
            public boolean hasNext() {
                return true;
            }

            @Override
            public DataSet next() {
                cnt++;
                return expectedDs.get(cnt);
            }
        }, expectedDs.size());

        assertTrue("Expect to be loaded", cache.isLoaded());
        assertTrue("Expect to have next", cache.hasNext());

        final List<DataSet> actualDs = new ArrayList<>();
        while (cache.hasNext()) {
            actualDs.add(cache.next());
        }

        assertEquals("Incorrect data!", expectedDs, actualDs);

        cache.resetCursor();
        final List<DataSet> actualDsAgain = new ArrayList<>();
        while (cache.hasNext()) {
            actualDsAgain.add(cache.next());
        }

        assertEquals("Incorrect data!", expectedDs, actualDsAgain);

        cache.clear();
        assertFalse("Does not expect to be loaded", cache.isLoaded());
        assertFalse("Does not expect to have next", cache.hasNext());

        cache.resetCursor();
        assertFalse("Does not expect to be loaded", cache.isLoaded());
        assertFalse("Does not expect to have next", cache.hasNext());


    }

}
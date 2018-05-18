package ampcontrol.model.training.data;

import org.junit.Test;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * Test cases for {@link ListShuffler}
 *
 * @author Christian Skärby
 */
public class ListShufflerTest {

    /**
     * Test that a shuffled list is provided
     */
    @Test
    public void apply() {
        List<Integer> testList = IntStream.range(0, 10).boxed().collect(Collectors.toList());
        List<Integer> result = new ListShuffler<Integer>(new Random(666)).apply(testList);
        assertTrue("Missing element!", result.containsAll(testList));
        assertNotEquals("List was not shuffled!", result, testList);
    }
}
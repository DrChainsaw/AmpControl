package ampcontrol.model.training.model.evolve.selection;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link EliteSelection}
 *
 * @author Christian Skärby
 */
public class EliteSelectionTest {

    /**
     * Test that the best two candidates are selected
     */
    @Test
    public void selectCandiates() {
        final double[] fitness = {666, 0.7, 13, 25};
        final int[] expected = {1, 2};

        final int[] selected = new EliteSelection<MockEvolvingItem>()
                .selectCandiates(MockEvolvingItem.createFitnessCands(fitness.length, i-> fitness[i]))
                .limit(expected.length)
                .map(MockEvolvingItem::toString)
                .mapToInt(Integer::parseInt)
                .toArray();

        assertArrayEquals("Incorrect selection!", expected, selected);
    }
}
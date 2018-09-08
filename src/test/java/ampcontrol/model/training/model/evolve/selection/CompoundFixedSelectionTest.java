package ampcontrol.model.training.model.evolve.selection;

import org.junit.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Testcases for {@link CompoundFixedSelection}
 *
 * @author Christian Skärby
 */
public class CompoundFixedSelectionTest {

    /**
     * Test selection of candidates
     */
    @Test
    public void selectCandiates() {
        final List<Map.Entry<Double, MockEvolvingItem>> fitnessCands = MockEvolvingItem.createFitnessCands(3, i -> i + 1);
        final List<MockEvolvingItem> selected = CompoundFixedSelection.<MockEvolvingItem>builder()
                .andThen(4, list -> Stream.generate(() -> list.get(0).getValue()))
                .andThen(3, list -> Stream.generate(() -> list.get(1).getValue()))
                .andThen(4, list -> Stream.generate(() -> list.get(2).getValue()))
                .build()
                .selectCandiates(fitnessCands)
                .collect(Collectors.toList());

        assertEquals("Incorrect number of elements!", 4+3+4, selected.size());

        IntStream.range(0, 3).forEach(i ->
                assertEquals("Incorrect item at position " + i + "!", fitnessCands.get(0).getValue(), selected.get(i)));
        IntStream.range(4, 4+3).forEach(i ->
                assertEquals("Incorrect item at position " + i + "!", fitnessCands.get(1).getValue(), selected.get(i)));
        IntStream.range(7, 7+4).forEach(i ->
                assertEquals("Incorrect item at position " + i + "!", fitnessCands.get(2).getValue(), selected.get(i)));

    }
}
package ampcontrol.model.training.model.evolve.selection;

import org.junit.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static junit.framework.TestCase.assertEquals;

public class EvolveSelectionTest {

    /**
     * Test that candidates selected by the source {@link Selection} are evolved.
     */
    @Test
    public void selectCandiates() {
        final List<Map.Entry<Double, MockEvolvingItem>> fitnessCands = MockEvolvingItem.createFitnessCands(5, i-> i+1);
        final List<String> evolved = new EvolveSelection<MockEvolvingItem>(list -> list.stream()
                .map(Map.Entry::getValue))
                .selectCandiates(fitnessCands)
                .map(MockEvolvingItem::toString)
                .collect(Collectors.toList());

        final List<String> expected = fitnessCands.stream()
                .map(Map.Entry::getValue)
                .map(MockEvolvingItem::evolvedName)
                .collect(Collectors.toList());

        assertEquals("Candidates were not evolved!", expected, evolved);
    }
}
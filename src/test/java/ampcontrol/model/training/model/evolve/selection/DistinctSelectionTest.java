package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.util.CompGraphUtil;
import org.junit.Test;

import java.util.Collections;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link DistinctSelection}
 *
 * @author Christian Skärby
 */
public class DistinctSelectionTest {

    /**
     * Test that selected candidates are distinct
     */
    @Test
    public void selectCandidates() {
        assertEquals("Incorrect output!", "a b c d e f",
                new DistinctSelection<String, String>(Function.identity(), cands -> Stream.of("a b c a c d e b f".split(" ")))
                        .selectCandiates(Collections.emptyList()).collect(Collectors.joining(" ")));
    }


    /**
     * Test distinct candidates from multiple {@link Selection}s
     */
    @Test
    public void selectDistinctCandiatesFromMultipleSelections() {
        final Selection<String> selection1 = cands -> Stream.of("Candidates from the first selection".split(" "));
        final Selection<String> selection2 = cands -> Stream.of(", second and".split(" "));
        final Selection<String> selection3 = cands -> Stream.of("Candidates from the third selection".split(" "));

        final DistinctSelection.Builder<String, String> distinctBuilder = DistinctSelection.builder(Function.identity());
        final Selection<String> selection =
                distinctBuilder.source(
                        CompoundSelection.<String>builder()
                                .andThen(distinctBuilder.distinct(new Limit<>(() -> 4, selection1)))
                                .andThen(distinctBuilder.distinct(selection2))
                                .andThen(distinctBuilder.distinct(selection3))
                                .build())
                        .build();

        assertEquals("Incorrect output!", "Candidates from the first , second and third selection",
                selection.selectCandiates(Collections.emptyList()).collect(Collectors.joining(" ")));

        // Try again to verify that state is cleared
        assertEquals("Incorrect output!", "Candidates from the first , second and third selection",
                selection.selectCandiates(Collections.emptyList()).collect(Collectors.joining(" ")));
    }

    /**
     * Test that ComputationGraphs can be seen as distinct
     */
    @Test
    public void distinctCompGraphs() {
        assertEquals("Incorrect output!", 2,
                new DistinctSelection<>(
                        CompGraphUtil::configUniquenessString,
                        cands -> Stream.of(
                                GraphUtils.getDoubleForkResNet("0", "1", "f0", "f1"),
                                GraphUtils.getDoubleForkResNet("0", "1", "f0", "f1", "f3"),
                                GraphUtils.getDoubleForkResNet("0", "1", "f0", "f1")
                        ))
                        .selectCandiates(Collections.emptyList()).count());

    }
}
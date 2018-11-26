package ampcontrol.model.training.model.evolve.crossover.state;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link GenericCrossoverState}
 *
 * @author Christian Skärby
 */
public class GenericCrossoverStateTest {

    /**
     * Test simple crossover with some "state" as input
     */
    @Test
    public void cross() {
        CrossoverState<String, String> crossoverState = createStringStateCrossover();
        assertEquals("Incorrect result!", "test has passed",
                crossoverState.cross("test", "passed", "has", "not"));
    }

    /**
     * Test that state is merged when doing crossover
     */
    @Test
    public void crossMerge() {
        final List<String> state1 = Stream.of("The", "fox", "swims", "across", "a", "lake").collect(Collectors.toCollection(ArrayList::new));
        final List<String> state2 = Arrays.asList("The", "flea", "jumps", "over", "the", "dog");

        final CrossoverState<List<String>, List<String>> crossover = createListStateCrossover(
                4, 3);

        final List<String> input1 = Stream.of("The", "quick", "brown", "fox", "swims", "across", "a", "lake").collect(Collectors.toList());
        final List<String> input2 = Stream.of("A", "tiny", "flea", "jumps", "over", "the", "lazy", "dog").collect(Collectors.toList());
        final List<String> result = crossover.cross(input1, input2, state1, state2);

        assertEquals("Incorrect result!",
                Arrays.asList("The", "quick", "brown", "fox","jumps", "over", "the", "lazy", "dog"), result);

        assertEquals("Incorrect state!",
                Arrays.asList("The", "fox","jumps", "over", "the", "dog"), state1);

        assertEquals("State 2 shall not be modified!!",
                Arrays.asList("The", "flea", "jumps", "over", "the", "dog"), state2);

    }

    private static CrossoverState<String, String> createStringStateCrossover() {
        return new GenericCrossoverState<>(
                (state1, state2, input1, input2, result) -> {/* */},
                state -> (str1, str2) -> str1 + " " + state + " " + str2);
    }


    private static GenericCrossoverState<List<String>, List<String>> createListStateCrossover(
            int point1,
            int point2) {
        return new GenericCrossoverState<List<String>, List<String>>( // Compiler randomly fails if removed (?!)
                (state1, state2, input1, input2, result) -> {
                    final List<String> newState = Stream.concat(
                            input1.stream().filter(state1::contains).filter(result::contains),
                            input2.stream().filter(state2::contains).filter(result::contains)
                    ).collect(Collectors.toList());
                    state1.clear();
                    state1.addAll(newState);
                },
                state -> (set1, set2) -> Stream.concat(
                        set1.stream().limit(point1),
                        set2.stream().skip(point2))
                        .collect(Collectors.toList()));
    }
}
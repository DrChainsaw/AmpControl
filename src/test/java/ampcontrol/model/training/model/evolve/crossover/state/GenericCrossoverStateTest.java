package ampcontrol.model.training.model.evolve.crossover.state;

import ampcontrol.model.training.model.evolve.state.GenericState;
import org.apache.commons.lang3.mutable.Mutable;
import org.apache.commons.lang3.mutable.MutableObject;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.fail;

public class GenericCrossoverStateTest {

    @Test
    public void cross() {
        CrossoverState<String, ?> crossoverState = createStringStateCrossover(new MutableObject<>("has"));
        assertEquals("Incorrect result!", "test has passed", crossoverState.cross("test", "passed"));
    }

    @Test
    public void crossClone() {
        final Mutable<String> state = new MutableObject<>("has");
        CrossoverState<String, ?> crossoverState = createStringStateCrossover(state).clone();
        state.setValue("has not");
        assertEquals("Incorrect result!", "test has passed", crossoverState.cross("test", "passed"));
    }

    @Test
    public void crossMerge() {
        final List<String> state1 = Stream.of("The", "fox", "swims", "across", "a", "lake").collect(Collectors.toList());
        final List<String> state2 = Stream.of("The", "flea", "jumps", "over", "the", "dog", "may", "cause", "nausea").collect(Collectors.toList());
        final List<String> stateList = new ArrayList<>();

        final CrossoverState<List<String>, List<String>> cross1 = createListStateCrossover(
                state1,
                (str, state) -> stateList.addAll(state),
                4, 3);

        final CrossoverState<List<String>, List<String>> cross2 = createListStateCrossover(
                state2,
                (str, state) -> fail("Should not happen!"),
                0, 0);

        final List<String> input1 = Stream.of("The", "quick", "brown", "fox", "swims", "across", "a", "lake").collect(Collectors.toList());
        final List<String> input2 = Stream.of("A", "tiny", "flea", "jumps", "over", "the", "lazy", "dog").collect(Collectors.toList());
        final List<String> result = cross1.cross(input1, input2);

        assertEquals("Incorrect result!", Arrays.asList("The", "quick", "brown", "fox","jumps", "over", "the", "lazy", "dog"), result);

        cross1.merge(cross2, input1, input2, result);
        try {
            cross1.save("dummy");
            assertEquals("Incorrect state!", Arrays.asList("The", "fox","jumps", "over", "the", "dog"), stateList);
        } catch (IOException e) {
            fail("Unexpected exception!");
        }
    }

    private static CrossoverState<String, ?> createStringStateCrossover(Mutable<String> startState) {
        return new GenericCrossoverState<>(
                new GenericState<>(startState,
                        mutableString -> new MutableObject<>(mutableString.getValue()),
                        (str, state) -> {/* Ignore*/}),
                (state1, state2, input1, input2, result) -> {/* */},
                state -> (str1, str2) -> str1 + " " + state.getValue() + " " + str2);
    }


    private static GenericCrossoverState<List<String>, List<String>> createListStateCrossover(
            List<String> startState,
            GenericState.PersistState<List<String>> persistState,
            int point1,
            int point2) {
        return new GenericCrossoverState<List<String>, List<String>>( // Compiler randomly fails if removed (?!)
                new GenericState<>(
                        startState,
                        ArrayList::new,
                        persistState),
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
package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.CompGraphAdapter;
import ampcontrol.model.training.model.GraphModelAdapter;
import org.apache.commons.lang.mutable.MutableLong;
import org.apache.commons.lang3.mutable.Mutable;
import org.apache.commons.lang3.mutable.MutableObject;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertFalse;

/**
 * Test cases for {@link FixedAgeSelection}
 */
public class FixedAgeSelectionTest {

    /**
     * Test that aging works as expected
     */
    @Test
    public void selectCandiates() throws IOException {
        final List<String> population = Arrays.asList("first", "cand", "another cand", "etc");

        final Selection<String> selection = new FixedAgeSelection<>(
                str -> str, // Strings are already equal by value
                // Mutate all but the first
                cands -> Stream.concat(Stream.of(cands.get(0).getValue()), IntStream.range(1, cands.size())
                        .mapToObj(cands::get)
                        .map(Map.Entry::getValue)
                        .map(str -> str + " evolved")),
                3
        );

        final List<String> expectedFirst = Arrays.asList("first", "cand evolved", "another cand evolved", "etc evolved");
        final List<String> actualFirst = selection.selectCandiates(addFitness(population)).collect(Collectors.toList());
        assertEquals("Incorrect population!", expectedFirst, actualFirst);

        final List<String> second = selection.selectCandiates(addFitness(actualFirst)).collect(Collectors.toList());
        assertEquals("Incorrect number of candidates!", population.size(), second.size());

        final List<String> third = selection.selectCandiates(addFitness(second)).collect(Collectors.toList());
        assertEquals("Incorrect number of candidates!", population.size(), third.size());

        final List<String> fourth = selection.selectCandiates(addFitness(third)).collect(Collectors.toList());
        assertFalse("Old candidate was not removed!", fourth.contains("first"));
    }

    private static <T> List<Map.Entry<Double, T>> addFitness(List<T> population) {
        return population.stream().map(cand -> new AbstractMap.SimpleEntry<>(1d, cand)).collect(Collectors.toList());
    }

    /**
     * Test that byConfig works as expected.
     */
    @Test
    public void byConfig() throws IOException {
        final List<CompGraphAdapter> population = Arrays.asList(
                createAdapter(3, "a"),
                createAdapter(4, "a"),
                createAdapter(3, "b"),
                createAdapter(5, "b"));

        MutableLong newNout = new MutableLong(6);
        Mutable<String> newName = new MutableObject<>("c");
        final Selection<CompGraphAdapter> selection = FixedAgeSelection.byConfig(2,
                new HashMap<>(),
                // Keep the two first candidates and make two new ones
                cands -> Stream.concat(
                        cands.stream()
                                .limit(2)
                                .map(Map.Entry::getValue),
                        Stream.of(
                                createAdapter(newNout.longValue(), newName.getValue()),
                                createAdapter(newNout.longValue() + 1, newName.getValue() + "2")
                        )));

        List<CompGraphAdapter> selected = selection.selectCandiates(addFitness(population)).collect(Collectors.toList());
        assertEquals("Incorrect selection!", population.get(0), selected.get(0));
        assertEquals("Incorrect selection!", population.get(1), selected.get(1));
        assertFalse("Incorrect selection!", population.contains(selected.get(2)));
        assertFalse("Incorrect selection!", population.contains(selected.get(3)));

        // One of the new members (selected[3]) happens to get the same config as an existing config -> will be removed in next
        // selection round
        newNout.setValue(3);
        newName.setValue("a");
        selected = selection.selectCandiates(addFitness(selected)).collect(Collectors.toList());
        assertEquals("Incorrect selection!", population.get(0), selected.get(0));
        assertEquals("Incorrect selection!", population.get(1), selected.get(1));
        assertFalse("Incorrect selection!", population.contains(selected.get(2)));
        assertFalse("Incorrect selection!", population.contains(selected.get(3)));

        newNout.setValue(5);
        newName.setValue("f");
        selected = selection.selectCandiates(addFitness(selected)).collect(Collectors.toList());
        assertEquals("Clone shall not be selected!", 3, selected.size());
        assertFalse("Incorrect selection!", population.contains(selected.get(0)));
        assertFalse("Incorrect selection!", population.contains(selected.get(1)));
        assertFalse("Incorrect selection!", population.contains(selected.get(2)));

        assertEquals("Population size shall not decrease!", 4,  selection.selectCandiates(addFitness(selected)).count());
        assertEquals("Population size shall not decrease!", 4,  selection.selectCandiates(addFitness(selected)).count());
        assertEquals("Population size shall decrease!", 2,  selection.selectCandiates(addFitness(selected)).count());
        assertEquals("Population size shall not decrease!", 2,  selection.selectCandiates(addFitness(selected)).count());

        selected.add(createAdapter(3, "a")); // Shall not be removed, config is cleared from ageMap by now!
        assertEquals("Population size shall not decrease!", 3,  selection.selectCandiates(addFitness(selected)).count());


    }

    private static CompGraphAdapter createAdapter(long nOut, String layerName) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("0", new DenseLayer.Builder().nOut(nOut).build(), "input")
                .addLayer(layerName, new DenseLayer.Builder().nOut(3).build(), "0")
                .addLayer("output", new OutputLayer.Builder().nOut(1).build(), layerName)
                .setInputTypes(InputType.feedForward(4))
                .build());
        graph.init();
        return new GraphModelAdapter(graph);
    }
}
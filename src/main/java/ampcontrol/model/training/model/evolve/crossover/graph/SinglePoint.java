package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.crossover.Crossover;
import ampcontrol.model.training.model.evolve.mutate.util.ForwardOf;
import ampcontrol.model.training.model.evolve.mutate.util.Traverse;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;

import java.util.Comparator;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Simple {@link Crossover} for {@link GraphBuilder}s. To keep things reasonably simple, only vertices which are the
 * single vertex connecting all vertices above and below them are eligible for crossover.
 * <br><br>
 * Assumes the input types have been set for the inputs.
 *
 * @author Christian Skärby
 */
public final class SinglePoint implements Crossover<GraphInfo> {

    private final Supplier<PointSelection> selectionSupplier;

    public static class PointSelection {
        private final double relativeLocation;
        private final double distance;

        public PointSelection(double distance, double relativeLocation) {
            this.relativeLocation = relativeLocation;
            this.distance = distance;
        }

        private double locationTarget() {
            return relativeLocation;
        }

        private double distanceTarget() {
            return distance;
        }
    }

    public SinglePoint(Supplier<PointSelection> selectionSupplier) {
        this.selectionSupplier = selectionSupplier;
    }

    @Override
    public GraphInfo cross(
            GraphInfo bottom,
            GraphInfo top) {

        final PointSelection pointSelection = selectionSupplier.get();

        final Comparator<CrossoverPoint> byDistance = Comparator.comparingDouble(
                point -> Math.abs(pointSelection.distanceTarget() - point.distance()));
        final Comparator<CrossoverPoint> byBottomLocation = Comparator.comparingDouble(
                point -> Math.abs(pointSelection.locationTarget() - point.bottom().location()));
        final Comparator<CrossoverPoint> byTiebreaker = Comparator.comparingInt(point -> (point.bottom().name() + point.top().name()).hashCode());

        return findValidVertices(bottom)
                .flatMap(bottomVertex -> findValidVertices(top)
                        .filter(topVertex -> topVertex.type() == bottomVertex.type())
                        .map(topVertex -> new CrossoverPoint(bottomVertex, topVertex)))
                .min(byDistance.thenComparing(byBottomLocation).thenComparing(byTiebreaker))
                .map(CrossoverPoint::execute)
                .orElse(bottom); // Not a single valid crossover point found
    }

    private static Stream<VertexData> findValidVertices(GraphInfo info) {
        return info.builder().getVertices().keySet().stream()
                .filter(vertex -> isValid(vertex, info.builder()))
                .map(vertex -> new VertexData(vertex, info));

    }

    private static boolean isValid(String vertex, GraphBuilder builder) {
        // Vertices could perhaps work, but would just be a waste as they will typically only be connected to one input
        if (
                builder.getVertices().get(vertex) instanceof ElementWiseVertex
                        || builder.getVertices().get(vertex) instanceof MergeVertex
                        || builder.getNetworkOutputs().contains(vertex)
                        || builder.getNetworkInputs().contains(vertex)) {
            return false;
        }


        final List<String> flattened = new Traverse<>(new ForwardOf(builder)).children(vertex)
                .collect(Collectors.toList());
        flattened.add(vertex);
        //Vertex is not inside any type of skip connection if we find all inputs to each found vertex in the flattened graph
        return flattened.stream()
                .filter(vert -> !vert.equals(vertex))

                .allMatch(vert -> flattened.containsAll(builder.getVertexInputs().get(vert)));
    }
}

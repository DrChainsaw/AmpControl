package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.mutate.util.ForwardOf;
import ampcontrol.model.training.model.evolve.mutate.util.Traverse;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;

/**
 * Data about a vertex for crossover purposes.
 *
 * @author Christian Skärby
 */
class VertexData {
    private final String name;
    private final GraphInfo info;
    private final double relativeLocation;

    VertexData(String name, GraphInfo info) {
        this.name = name;
        this.info = info;
        this.relativeLocation = 1 - new Traverse<>(
                vert -> !vert.equals(name),
                new ForwardOf(builder()))
                .children(name).count() / (double) builder().getVertices().size();
        //  System.out.println("Valid vert " + name + " lcc " + relativeLocation);
    }

    InputType.Type type() {
        return builder().getLayerActivationTypes().get(name).getType();
    }

    ComputationGraphConfiguration.GraphBuilder builder() {
        return info.builder();
    }

    double location() {
        return relativeLocation;
    }

    GraphInfo info() {
        return info;
    }

    String name() {return name;}
}

package ampcontrol.model.training.model;

import ampcontrol.model.training.model.naming.FileNamePolicy;
import ampcontrol.model.training.model.validation.CachingValidationFactory;
import ampcontrol.model.training.model.validation.Validation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.TrainingListener;

import java.io.IOException;
import java.util.*;
import java.util.function.Function;

/**
 * A population of {@link ModelHandle}s fronted as a single {@link ModelHandle}. Typically used for architecture search.
 *
 * @author Christian Skärby
 */
public class ModelHandlePopulation implements ModelHandle {

    private ModelHandle bestModel;
    private final List<ModelHandle> population;
    private final Set<Validation.Factory<? extends IEvaluation>> validationFactories = new LinkedHashSet<>();
    private final Function<Integer, FileNamePolicy> candidateFileNamePolicy;
    private final String name;

    public ModelHandlePopulation(List<ModelHandle> population, String name, Function<Integer, FileNamePolicy> candidateFileNamePolicy) {
        this.population = population;
        this.name = name;
        this.candidateFileNamePolicy = candidateFileNamePolicy;
        bestModel = population.get(0);
    }

    @Override
    public void fit() {
        if (bestModel != population.get(0)) {
            bestModel = population.get(0);
            validationFactories.forEach(validationFactory -> bestModel.registerValidation(validationFactory));
        }
        population.forEach(ModelHandle::fit);
    }

    @Override
    public void eval() {
        // TODO: Real solution. Maybe getModel return ModelFacade and spy decorator here as well?
        final Collection<TrainingListener> listeners = new ArrayList<>(((ComputationGraph) population.get(0).getModel()).getListeners());
        ((ComputationGraph) population.get(0).getModel()).getListeners().clear();
        // Population list might be changed as a result of this operation
        new ArrayList<>(population).forEach(ModelHandle::eval);
        population.get(0).getModel().addListeners(listeners.toArray(new TrainingListener[0]));
    }

    @Override
    public void resetTraining() {
        population.forEach(ModelHandle::resetTraining);
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public Model getModel() {
        return population.get(0).getModel();
    }

    @Override
    public void registerValidation(Validation.Factory<? extends IEvaluation> validationFactory) {
        validationFactories.add(new CachingValidationFactory<>(validationFactory));
    }

    @Override
    public void saveModel(String fileName) throws IOException {
        for (int i = 0; i < population.size(); i++) {
            population.get(i).saveModel(candidateFileNamePolicy.apply(i).toFileName(fileName ));
        }
    }

}

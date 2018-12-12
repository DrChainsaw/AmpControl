package ampcontrol.model.training.model;

import ampcontrol.model.training.model.evolve.Population;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import ampcontrol.model.training.model.validation.CachingValidationFactory;
import ampcontrol.model.training.model.validation.Validation;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.evaluation.IEvaluation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A population of {@link ModelHandle}s fronted as a single {@link ModelHandle}. Typically used for architecture search.
 *
 * @author Christian Skärby
 */
public class ModelHandlePopulation implements ModelHandle {

    private final Population<ModelHandle> population;
    private final Set<Validation.Factory<? extends IEvaluation>> validationFactories = new LinkedHashSet<>();
    private final List<TrainingListener> listeners = new ArrayList<>();
    private final Function<Integer, FileNamePolicy> candidateFileNamePolicy;
    private final String name;
    private final Consumer<String> saveNameListener;

    public ModelHandlePopulation(
            Population<ModelHandle> population,
            String name,
            Function<Integer, FileNamePolicy> candidateFileNamePolicy,
            Consumer<String> saveNameListener) {
        this.population = population;
        this.name = name;
        this.candidateFileNamePolicy = candidateFileNamePolicy;
        this.saveNameListener = saveNameListener;

        // When population changes we need to add back listeners and validation since those are wiped at such change.
        population.onChangeCallback(() -> {
            validationFactories.forEach(validationFactory -> firstModel().registerValidation(validationFactory));
            listeners.forEach(listener -> firstModel().addListener(listener));
        });
    }

    @Override
    public void fit() {
        final List<ModelHandle> pop = population.streamPopulation().collect(Collectors.toList());
        for (int i = 0; i < pop.size(); i++) {
            try {
                pop.get(i).fit();
            } catch (Exception e) {
                throw new IllegalStateException("Failure while fitting model "  + i, e);
            }
        }
    }

    @Override
    public void eval() {
        population.streamPopulation().forEach(ModelHandle::eval);
    }

    @Override
    public void resetTraining() {
        population.streamPopulation().forEach(ModelHandle::resetTraining);
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public void registerValidation(Validation.Factory<? extends IEvaluation> validationFactory) {
        final Validation.Factory<? extends IEvaluation> cachingFactory = new CachingValidationFactory<>(validationFactory);
        validationFactories.add(cachingFactory);
        firstModel().registerValidation(cachingFactory);
    }

    @Override
    public void addListener(TrainingListener listener) {
        firstModel().addListener(listener);
        listeners.add(listener);
    }

    private ModelHandle firstModel() {
        return population.streamPopulation().findFirst().orElseThrow(() -> new IllegalStateException("No model found!"));
    }

    @Override
    public void saveModel(String fileName) throws IOException {
        final List<ModelHandle> pop = population.streamPopulation().collect(Collectors.toList());
        for (int i = 0; i < pop.size(); i++) {
            pop.get(i).saveModel(candidateFileNamePolicy.apply(i).toFileName(fileName));
        }
        saveNameListener.accept(fileName);
    }
}

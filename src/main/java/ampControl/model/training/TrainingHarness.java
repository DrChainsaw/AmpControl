package ampControl.model.training;

import ampControl.model.training.listen.IterationSupplier;
import ampControl.model.training.listen.TrainScoreListener;
import ampControl.model.training.model.ModelHandle;
import ampControl.model.training.model.validation.EvalValidation;
import ampControl.model.training.model.validation.Skipping;
import ampControl.model.training.model.validation.Validation;
import ampControl.model.training.model.validation.listen.*;
import ampControl.model.visualize.Plot;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * Harness for training and evaluation of {@link ModelHandle ModelHandles}. Models will take turn in doing fitting. Will
 * evaluate models in regular intervals and saves the last evaluated model as long as the accuracy is not worse than 90%
 * of the best evaluation accuracy for the model. Will also plot accuracy and score for training, eval and best eval.
 *
 * @author Christian Skärby
 */
class TrainingHarness {

    private static final Logger log = LoggerFactory.getLogger(TrainingHarness.class);

    private static final boolean doStatsLogging = false;
    private static final int evalEveryNrofSteps = 4;
    private static final String bestSuffix = "_best";
    private static final double saveThreshold = 0.9;

    private static final String trainEvalPrefix = "Train";
    private static final String lastEvalPrefix = "LastEval";
    private static final String bestEvalPrefix = "BestEval";

    private final List<ModelHandle> modelsToTrain;
    private final String modelSaveDir;
    private final Plot.Factory<Integer, Double> plotFactory;

    TrainingHarness(List<ModelHandle> modelsToTrain, String modelSaveDir, Plot.Factory<Integer, Double> plotFactory) {
        this.modelsToTrain = modelsToTrain;
        this.modelSaveDir = modelSaveDir;
        this.plotFactory = plotFactory;
    }

    private void addListeners(final List<ModelHandle> models) {

        for (final ModelHandle md : models) {
            if (doStatsLogging) {
                final UIServer uiServer = UIServer.getInstance();
                //Alternative: new FileStatsStorage(File) - see UIStorageExample
                final StatsStorage statsStorage = new FileStatsStorage(new File(md.name() + "_stats"));
                uiServer.attach(statsStorage);
                final int listenerFrequency = md.getNrofBatchesForTraining();
                md.getModel().addListeners(new StatsListener(statsStorage, listenerFrequency));
            }
            md.getModel().addListeners(new ScoreIterationListener(md.getNrofBatchesForTraining()));
        }
    }

    private void addValidation(final List<ModelHandle> models) {
        final Plot<Integer, Double> evalPlot = initPlot("Accuracy", models);
        final Plot<Integer, Double> scorePlot = initPlot("Score", models);

        for (ModelHandle mh : models) {
            final IterationSupplier iterationSupplier = new IterationSupplier();
            mh.getModel().addListeners(iterationSupplier);

            final String trainName = trainEvalName(mh.name());
            mh.getModel().addListeners(new TrainScoreListener(mh.getNrofBatchesForTraining(), (i, s) -> scorePlot.plotData(trainName, i, s)));
            mh.createTrainingEvalListener((i, e) -> evalPlot.plotData(trainName, i, e));

            final Validation.Factory<Evaluation> evaluationValidation = (labels -> {
                final Consumer<Evaluation> listener = createEvalConsumer(mh, iterationSupplier, evalPlot, scorePlot)
                        .andThen(createLastCheckPoint(mh))
                        .andThen(createBestCheckPoint(mh, iterationSupplier, evalPlot, scorePlot));


                return new Skipping<>(
                        new Skipping<>(
                                new EvalValidation(new Evaluation(labels), listener),
                                eval -> (int) Math.floor(10 * (1 - eval.accuracy()))), // TODO: Break out and test?
                        eval -> evalEveryNrofSteps, 2);
            });
            mh.registerValidation(evaluationValidation);
        }
    }

    private Plot<Integer, Double> initPlot(final String title, final List<ModelHandle> models) {
        final Plot<Integer, Double> plot = plotFactory.create(title);
        for (ModelHandle md : models) {
            plot.createSeries(trainEvalName(md.name()));
            plot.createSeries(lastEvalName(md.name()));
            plot.createSeries(bestEvalName(md.name()));
        }
        return plot;
    }

    private Consumer<Evaluation> createEvalConsumer(final ModelHandle model,
                                                    final Supplier<Integer> iterationSupplier,
                                                    final Plot<Integer, Double> evalPlot,
                                                    final Plot<Integer, Double> scorePlot) {
        final String lastEvalLabel = lastEvalName(model.name());
        final Consumer<Evaluation> plotEval = eval -> evalPlot.plotData(lastEvalLabel, iterationSupplier.get(), eval.accuracy());
        final Consumer<Evaluation> plotScore = eval -> scorePlot.plotData(lastEvalLabel, iterationSupplier.get(), model.getModel().score());

        final Consumer<Evaluation> storePlots =
                eval -> {
                    try {
                        evalPlot.storePlotData(trainEvalName(model.name()));
                        evalPlot.storePlotData(lastEvalLabel);
                        scorePlot.storePlotData(trainEvalName(model.name()));
                        scorePlot.storePlotData(lastEvalLabel);
                    } catch (IOException e) {
                        log.warn(e.getMessage());
                    }
                };

        final EvalLog evalLog = new EvalLog(model.name(), model.getBestEvalScore());

        return new NewThread<>( // Background work
                new Synced<>( // To avoid mixed up loggig
                        this,
                        evalLog
                                .andThen(plotEval)
                                .andThen(plotScore)
                                .andThen(storePlots)));
    }

    private Consumer<Evaluation> createLastCheckPoint(final ModelHandle model) {
        final String fileBaseName = modelSaveDir + File.separator + model.name().hashCode();
        final Consumer<Evaluation> saveCheckPoint = createCheckPoint(model, fileBaseName);


        final Predicate<Evaluation> gate = new Predicate<Evaluation>() {
            private double bestAccuracy = model.getBestEvalScore();

            @Override
            public boolean test(Evaluation eval) {
                boolean ok = eval.accuracy() >= saveThreshold * bestAccuracy;
                bestAccuracy = Math.max(eval.accuracy(), bestAccuracy);
                return ok;
            }
        };

        return new NewThread<>( // Background work
                new Gated<>(saveCheckPoint, gate));
    }

    private Consumer<Evaluation> createBestCheckPoint(final ModelHandle model,
                                                      final Supplier<Integer> iterationSupplier,
                                                      final Plot<Integer, Double> evalPlot,
                                                      final Plot<Integer, Double> scorePlot) {
        final String fileBaseName = modelSaveDir + File.separator + model.name().hashCode() + bestSuffix;
        final Consumer<Evaluation> saveCheckPoint = createCheckPoint(model, fileBaseName);

        final String bestEvalLabel = bestEvalName(model.name());
        final Consumer<Evaluation> plotEval = eval -> evalPlot.plotData(bestEvalLabel, iterationSupplier.get(), eval.accuracy());
        final Consumer<Evaluation> plotScore = eval -> scorePlot.plotData(bestEvalLabel, iterationSupplier.get(), model.getModel().score());

        final Consumer<Evaluation> storePlots =
                eval -> {
                    try {
                        evalPlot.storePlotData(bestEvalLabel);
                        scorePlot.storePlotData(bestEvalLabel);
                    } catch (IOException e) {
                        log.warn(e.getMessage());
                    }
                };


        final Predicate<Evaluation> gate = new Predicate<Evaluation>() {
            private double bestAccuracy = model.getBestEvalScore();

            @Override
            public boolean test(Evaluation eval) {
                boolean ok = eval.accuracy() >= bestAccuracy;
                bestAccuracy = Math.max(eval.accuracy(), bestAccuracy);
                return ok;
            }
        };

        return new NewThread<>( // Background work
                new Gated<>(saveCheckPoint
                        .andThen(plotEval)
                        .andThen(plotScore)
                        .andThen(storePlots), gate));
    }

    private Consumer<Evaluation> createCheckPoint(final ModelHandle model, final String fileBaseName) {
        final Consumer<Evaluation> scoreCheckPoint = new EvalCheckPoint(fileBaseName, model.name());
        final Consumer<Evaluation> modelCheckPointEc = eval -> {
            try {
                model.saveModel(fileBaseName);
            } catch (IOException e) {
                log.warn(e.getMessage());
            }
        };
        return modelCheckPointEc.andThen(scoreCheckPoint);
    }

    /**
     * Trains the models
     *
     * @param maxNrofTrainingSteps number of training steps
     */
    void startTraining(final int maxNrofTrainingSteps) {

        addListeners(modelsToTrain);
        addValidation(modelsToTrain);

        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        for (int trainingStep = 0; trainingStep < maxNrofTrainingSteps; trainingStep++) {
            log.info("****************************** Training step " + trainingStep + " started! ***************************************");
            for (ModelHandle mh : modelsToTrain) {
                mh.fit();
            }

            for (ModelHandle mh : modelsToTrain) {
                mh.eval();
                mh.resetTraining();
            }
        }
    }

    private String trainEvalName(String modelName) {
        return trainEvalPrefix + modelName.hashCode();
    }

    private String lastEvalName(String modelName) {
        return lastEvalPrefix + modelName.hashCode();
    }

    private String bestEvalName(String modelName) {
        return bestEvalPrefix + modelName.hashCode();
    }
}

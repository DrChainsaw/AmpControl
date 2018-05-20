package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.data.iterators.preprocs.CnnToRnnToLastStepToFfPreProcessor;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.layerblocks.Dense;
import ampcontrol.model.training.model.layerblocks.LstmBlock;
import ampcontrol.model.training.model.layerblocks.Output;
import ampcontrol.model.training.model.layerblocks.RnnType;
import ampcontrol.model.training.model.layerblocks.graph.LastStep;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Pure LSTM model. Have not gotten any good performance out of it which is not unexptected.
 *
 * @author Christian Skärby
 */
public class LstmTimeSeqFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public LstmTimeSeqFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelDir = modelDir;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {
        IntStream.of(1, 2, 4).forEach(nrofLstmLayers -> {
            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                    new BlockBuilder()
                            .setUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 0.05, 0.1, 40000)))
                            .setNamePrefix(namePrefix)
                            .first(new RnnType(inputShape))//.setTbpttLength(2000)
                            .andThenStack(nrofLstmLayers)
                            .of(new LstmBlock()
                                    .setWidth(256))
                            .andThen(new LastStep())
                            .andThenStack(2)
                            .of(new Dense())
                            .andFinally(new Output(trainIter.totalOutcomes())));

            modelData.add(new GenericModelHandle(
                    trainIter,
                    evalIter,
                    new ModelAdapterWithPreProc(new CnnToRnnToLastStepToFfPreProcessor(),
                            new GraphModelAdapter(builder.buildGraph())),
                    builder.name()));

        });
    }
}

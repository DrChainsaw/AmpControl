package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.data.iterators.preprocs.Cnn2DtoCnn1DInputPreprocessor;
import ampcontrol.model.training.model.GenericModelHandle;
import ampcontrol.model.training.model.GraphModelAdapter;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.GlobMeanMax;
import ampcontrol.model.training.model.layerblocks.graph.PreprocVertex;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.util.List;
import java.util.stream.DoubleStream;

/**
 * Description of sound part of soundnet from https://arxiv.org/abs/1610.09001.
 *
 * @author Christian Skärby
 */
public class SoundnetFactory {
    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final FileNamePolicy modelFileNamePolicy;

    public SoundnetFactory(MiniEpochDataSetIterator trainIter, MiniEpochDataSetIterator evalIter, int[] inputShape, String namePrefix, FileNamePolicy modelFileNamePolicy) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelFileNamePolicy = modelFileNamePolicy;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {
        //Soundnet
        DoubleStream.of(0).forEach(dropOutProb -> {
            ModelBuilder builder = new DeserializingModelBuilder(modelFileNamePolicy,
                    new BlockBuilder()
                            .setNamePrefix(namePrefix)
                            .setUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 0.001, 0.1, 40000)))
                            .first(new ConvTimeType(inputShape))
                            .andThen(new PreprocVertex()
                                    .setPreProcessor(new Cnn2DtoCnn1DInputPreprocessor()))

                            // First block
                            .andThen(new ZeroPad1D().setPadding(32))
                            .andThen(new Conv1D()
                                    .setNrofKernels(16)
                                    .setKernelSize(64)
                                    .setStride(2))
                            // .setInputPreproc(new Cnn2DtoCnn1DInputPreprocessor()))
                            .andThen(new Pool1D().setSize(8).setStride(1))

                            //Second block
                            .andThen(new ZeroPad1D().setPadding(16))
                            .andThen(new Conv1D()
                                    .setNrofKernels(32)
                                    .setKernelSize(32)
                                    .setStride(2))
                            .andThen(new Pool1D().setSize(8).setStride(1))

                            //Third block
                            .andThen(new ZeroPad1D().setPadding(8))
                            .andThen(new Conv1D()
                                    .setNrofKernels(64)
                                    .setKernelSize(16)
                                    .setStride(2))

                            //4:th block
                            .andThen(new ZeroPad1D().setPadding(4))
                            .andThen(new Conv1D()
                                    .setNrofKernels(128)
                                    .setKernelSize(8)
                                    .setStride(2))

                            //5:th block
                            .andThen(new ZeroPad1D().setPadding(2))
                            .andThen(new Conv1D()
                                    .setNrofKernels(256)
                                    .setKernelSize(4)
                                    .setStride(2))
                            .andThen(new Pool1D().setSize(4).setStride(1))

                            // 6:th block
                            .andThen(new ZeroPad1D().setPadding(2))
                            .andThen(new Conv1D()
                                    .setNrofKernels(512)
                                    .setKernelSize(4)
                                    .setStride(2))

                            // 7:th block
                            .andThen(new ZeroPad1D().setPadding(2))
                            .andThen(new Conv1D()
                                    .setNrofKernels(1024)
                                    .setKernelSize(4)
                                    .setStride(2))

                            // 8:th block
                            .andThen(new Conv1D()
                                    .setNrofKernels(1401)
                                    .setKernelSize(8)
                                    .setStride(2))

                            //.andThen(new Dense())
                            .andThen(new GlobMeanMax())
                            .andThen(new DropOut().setDropProb(dropOutProb))
                            .andFinally(new Output(trainIter.totalOutcomes())));

            modelData.add(new GenericModelHandle(
                    trainIter,
                    evalIter,
                    new GraphModelAdapter(builder.buildGraph()),
                    builder.name()));
        });
    }
}

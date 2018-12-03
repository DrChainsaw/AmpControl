package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.model.GenericModelHandle;
import ampcontrol.model.training.model.GraphModelAdapter;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.util.List;
import java.util.stream.DoubleStream;

/**
 * Description of a bunch of architectures which belong to a family of shallow but wide 2D convolutional neural networks.
 * Idea comes from some arxiv paper, but I can't remember which one. Seems to perform worse than 2D CNNs with small
 * kernel sizes.
 *
 * @author Christian Skärby
 */
public class Conv2DShallowWideFactory {
    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final FileNamePolicy modelFileNamePolicy;

    public Conv2DShallowWideFactory(MiniEpochDataSetIterator trainIter, MiniEpochDataSetIterator evalIter, int[] inputShape, String namePrefix, FileNamePolicy modelFileNamePolicy) {
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

        DoubleStream.of(0).forEach(dropOutProb -> {
            int kernelSizeLong = inputShape[1] - 2; // Allow for 2 frequency bins invariance
            // int kernelSizeHalf = inputShape[1] / 2;
            int poolSizeTime = inputShape[0] / 2;
            ModelBuilder builder = new DeserializingModelBuilder(modelFileNamePolicy,
                    new BlockBuilder()
                            .setUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 0.0005, 0.1, 40000)))
                            .setNamePrefix(namePrefix)
                            .first(new ConvType(inputShape))
                            //.multiLevel()
                            .andThen(new Conv2DBatchNormAfter()
                                    .setKernelSize_w(kernelSizeLong)
                                    .setKernelSize_h(2)
                                    .setNrofKernels(128))
//                    .andThen(new Conv2DBatchNormAfter()
//                            .setKernelSize_w(kernelSizeHalf)
//                            .setKernelSize_h(2)
//                            .setStride_w(kernelSizeHalf / 8)
//                            .setNrofKernels(128))
                            //  .done()
                            //.andThen(new DropOut().setDropProb(dropOutProb))
                            .andThen(new Pool2D().setSize_h(poolSizeTime).setSize_w(1).setStride_h(poolSizeTime / 8).setSize_w(1))
                            .andThen(new Conv2DBatchNormAfter()
                                    .setKernelSize_w(3)
                                    .setKernelSize_h(3)
                                    .setNrofKernels(128))
                            //.andThen(new GlobMeanMax())
                            .andThen(new Dense()
                                    .setHiddenWidth(256)
                                    .setActivation(new ActivationReLU()))
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

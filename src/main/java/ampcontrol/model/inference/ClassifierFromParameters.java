package ampcontrol.model.inference;

import ampcontrol.audio.ClassifierInputProvider;
import ampcontrol.audio.ClassifierInputProviderFactory;
import com.beust.jcommander.Parameter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Factory for creating a {@link Classifier}. Parameterized through {@link com.beust.jcommander.JCommander}.
 *
 * @author Christian Skärby
 */
public class ClassifierFromParameters {

	private final static String modelBaseDir = "E:\\Software projects\\java\\leadRythm\\RythmLeadSwitch\\models\\";

	@Parameter(names = "-classifiers", description = "Comma separated list of paths to classifiers to use. If more than " +
			"one, an ensemble model will be created.")
	private List<String> classifiers = Arrays.asList(
			//modelBaseDir + "ws_100_sgpp_spgr_fft_256_olf_8_pipe_lgsc_4x_C_128_3_ELU_BN_w_Mp2_2_t_2x_dnn_512_ReLU_w_d0p0_t_out_w_0p5_0p5_0p8_1p0Nesterovs_best",
			//modelBaseDir + "ws_100_sgpp_umzm_pp_t_zp2_1_0_t_C_256_3_1_ELU_BN_t_se0.0625_ReLU_t_2x_rb_2x_zp2_1_0_w_C_256_3_1_ELU_BN_t_se0.0625_ReLU_w_Mp3_1_3_1_t_mla_rb_2x_zp2_1_0_w_C_256_3_1_ELU_BN_t_se0.0625_ReLU_t_Mp3_1_3_1_t_rb_2x_zp2_1_0_w_C_256_3_1_ELU_BN_t_se0.0625_ReLU_t_Mp3_1_3_1_t_2x_zp2_1_0_w_C_512_3_1_ELU_BN_t_se0.0625_ReLU_t_rb_2x_zp2_1_0_w_C_512_3_1_ELU_BN_t_se0.0625_ReLU_t_Mp3_1_3_1_alm_t_2x_dnn_256_ReLU_w_d0p0_t_out_w_0p5_0p5_0p8_1p0Adam_best",
			//modelBaseDir + "ws_100_sgpp_spgr_fft_256_olf_8_pipe_lgsc_C_64_3_ELU_BN_t_C_64_3_ELU_BN_t_Mp2_2_t_C_128_3_ELU_BN_t_C_128_3_ELU_BN_t_Mp2_2_t_C_256_3_ELU_BN_t_C_256_3_ELU_BN_t_Mp2_2_t_C_512_3_ELU_BN_t_C_512_3_ELU_BN_t_Mp2_2_t_2x_dnn_512_ReLU_w_d0p0_t_out_w_0p5_0p5_0p8_1p0Nesterovs_best",
			//modelBaseDir + "ws_100_sgpp_spgr_fft_256_olf_8_pipe_lgsc_C_128_3_ELU_BN_t_4x_C_128_3_ELU_BN_w_Mp2_2_t_2x_dnn_512_ReLU_w_d0p0_t_out_w_0p5_0p5_0p8_1p0Nesterovs_best",
			//modelBaseDir + "ws_100_sgpp_spgr_fft_256_olf_8_pipe_lgsc_zp2_5_5_t_C_32_6_ELU_BN_t_Mp3_2_t_3x_3rs_C_64_3_ELU_BN_w_zp2_1_1_w_C_96_1_ELU_BN_t_Ap2_2_t_3rs_C_128_3_ELU_BN_w_zp2_1_1_t_2x_dnn_256_ReLU_t_d0p0_t_out_w_0p5_0p5_0p8_1p0Nesterovs_best"
			modelBaseDir + "ws_50_sgpp_spgr_fft_256_olf_16_pipe_fork_lg10_pipe_zm_split_mfsc_pipe_zm_krof_C_64_3_ELU_BN_t_C_64_3_ELU_BN_t_Mp2_2_t_C_128_3_ELU_BN_t_C_128_3_ELU_BN_t_Mp2_2_t_C_256_3_ELU_BN_t_C_256_3_ELU_BN_t_Mp2_2_t_C_512_3_ELU_BN_t_C_512_3_ELU_BN_t_Mp2_2_t_2x_dnn_512_ReLU_w_d0p0_t_out_w_0p5_0p5_0p8_1p0Nesterovs_best",
			modelBaseDir + "ws_50_sgpp_spgr_fft_256_olf_16_pipe_fork_lg10_pipe_zm_split_mfsc_pipe_zm_krof_zp2_5_5_t_C_32_6_ELU_BN_t_Mp3_2_t_3x_3ds_C_64_3_ELU_BN_w_zp2_1_1_w_C_96_1_ELU_BN_t_Ap2_2_t_3ds_C_128_3_ELU_BN_w_zp2_1_1_t_2x_dnn_512_ReLU_t_d0p0_t_out_w_0p5_0p5_0p8_1p0Nesterovs_best"
			//modelBaseDir + "ws_200_fft_1024_olf_16_nfs_128_Cnn2d_4_layers_128_kernels_4_4_mp_2_2_dnn_2_layers_64_dnnW_best"
			//modelBaseDir + "ws_100_fft_1024_olf_32_nfs_128_Cnn2d_4_layers_128_kernels_4_4_mp_2_2_dnn_2_layers_64_dnnW_best"
			//modelBaseDir + "ws_200_fft_1024_olf_16_nfs_128_LstmTimeSeq_2_layers_64_lstmW_best"
			//,modelBaseDir + "ws_200_fft_1024_olf_16_nfs_128_LstmTimeSeq_4_layers_128_lstmW_best"
			// ,modelBaseDir + "ws_200_fft_1024_olf_16_nfs_128_Cnn2d_2_layers_64_kernels_3_3_mp_4_4_dnn_3_layers_64_dnnW_best"
			// ,modelBaseDir + "ws_200_fft_1024_olf_16_nfs_128_Cnn2d_4_layers_128_kernels_4_4_mp_2_2_dnn_4_layers_64_dnnW_best"
			// ,modelBaseDir + "ws_200_fft_1024_olf_16_nfs_128_Cnn2d_4_layers_128_kernels_3_3_mp_2_2_dnn_2_layers_64_dnnW_best"
			//,modelBaseDir + "ws_100_fft_512_ol_16_Cnn2d_4_layers_128_kernels_2_2_mp_2_2_dnn_2_layers_64_dnnW"
			//  ,modelBaseDir + "ws_100_fft_512_ol_16_Cnn2d_2_layers_16_kernels_12_12_mp_2_2_dnn_2_layers_256_dnnW_best"
			//  ,modelBaseDir + "ws_100_fft_512_ol_16_Cnn2d_4_layers_16_kernels_6_6_mp_2_2_dnn_1_layers_256_dnnW_best"
			);

	@Parameter(names = {"-classificationForgettingFactor", "-cff"}, description = "Set forgetting factor for classification")
	private double cff = 0.85;

	// Make parameter if ever needed? Or maybe needs to figure out type from name?
	private ClassifierFactory factory = (name, inputProvider) -> new StoredGraphClassifier(name, inputProvider);

	/**
	 * Creates a {@link Classifier}.
	 *
	 * @param inputFactory Factory for creating input to the classifier
	 * @return a {@link Classifier}
	 * @throws IOException
	 */
	public Classifier getClassifier(ClassifierInputProviderFactory inputFactory) throws IOException {

		List<Classifier> classifierList = new ArrayList<>();
		for(String classifierName: classifiers) {
            final ClassifierInputProvider inputProvider = inputFactory.createInputProvider(classifierName);
			classifierList.add(factory.create(classifierName, inputProvider));
		}

		Classifier ret;
		if(classifierList.size() > 0) {
			ret = new EnsembleWeightedSumClassifier(classifierList, EnsembleWeightedSumClassifier.avgNormalizer);
		} else {
			ret = classifierList.get(0);
		}
		//ret = new SpyClassifier(ret, inputProvider, new SpyClassifier.PlotSpecgramListener(), 5, 2,1);

		if(cff < 1) {
			ret = new EwmaClassifier(cff, ret);
		}


		return ret;
	}

	/**
	 * Sets the factory for creating classifiers from classifier name. Intended use is testing.
	 *
	 * @param factory
	 */
	void setFactory(ClassifierFactory factory) {
		this.factory = factory;
	}
}

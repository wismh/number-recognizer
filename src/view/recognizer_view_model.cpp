#include "view/recognizer_view_model.h"

namespace game {

RecognizerViewModel::RecognizerViewModel() {
    using engine::ui::intern;

    property(intern("statusText"), statusText);
    property(intern("tabRecognize"), tabRecognize);
    property(intern("tabNeurons"), tabNeurons);
    property(intern("tabTrain"), tabTrain);
    property(intern("tabDatasets"), tabDatasets);
    property(intern("tabHelp"), tabHelp);
    property(intern("localeIsEn"), localeIsEn);
    property(intern("localeIsUk"), localeIsUk);
    command(intern("localeEn"), localeEn);
    command(intern("localeUk"), localeUk);
    command(intern("showHelp"), showHelp);
    showHelp.bind_to<RecognizerViewModel, &RecognizerViewModel::show_help>(*this);
    command(intern("showRecognize"), showRecognize);
    command(intern("showNeurons"), showNeurons);
    command(intern("showTrain"), showTrain);
    command(intern("showDatasets"), showDatasets);
    showRecognize.bind_to<RecognizerViewModel, &RecognizerViewModel::show_recognize>(*this);
    showNeurons.bind_to<RecognizerViewModel, &RecognizerViewModel::show_neurons>(*this);
    showTrain.bind_to<RecognizerViewModel, &RecognizerViewModel::show_train>(*this);
    showDatasets.bind_to<RecognizerViewModel, &RecognizerViewModel::show_datasets>(*this);

    property(intern("predictionText"), predictionText);
    property(intern("modelLoaded"), modelLoaded);
    property(intern("modelMissing"), modelMissing);
    property(intern("modelNameText"), modelNameText);
    property(intern("modelArchText"), modelArchText);
    property(intern("modelLayersText"), modelLayersText);
    property(intern("modelEpochsText"), modelEpochsText);
    property(intern("testText"), testText);
    property(intern("testMutatedText"), testMutatedText);
    property(intern("hardestDigitText"), hardestDigitText);
    property(intern("easiestDigitText"), easiestDigitText);
    property(intern("perDigitText"), perDigitText);
    property(intern("addLabelText"), addLabelText);
    property(intern("addTargetText"), addTargetText);
    command(intern("clearCanvas"), clearCanvas);
    command(intern("addLabelUp"), addLabelUp);
    command(intern("addLabelDown"), addLabelDown);
    command(intern("addSample"), addSample);
    paint(intern("canvasPaint"), canvasPaint);
    paint(intern("probabilityPaint"), probabilityPaint);
    paint(intern("prototypePaint"), prototypePaint);

    property(intern("neuronsLayerText"), neuronsLayerText);
    property(intern("neuronsProgressText"), neuronsProgressText);
    property(intern("neuronTitleText"), neuronTitleText);
    property(intern("neuronLegendText"), neuronLegendText);
    property(intern("neuronBiasText"), neuronBiasText);
    property(intern("neuronNowText"), neuronNowText);
    property(intern("neuronInfluenceText"), neuronInfluenceText);
    property(intern("hasInfluence"), hasInfluence);
    property(intern("weightsActive"), weightsActive);
    property(intern("weightsIdle"), weightsIdle);
    property(intern("dreamActive"), dreamActive);
    property(intern("dreamIdle"), dreamIdle);
    command(intern("neuronsLayerUp"), neuronsLayerUp);
    command(intern("neuronsLayerDown"), neuronsLayerDown);
    command(intern("showWeights"), showWeights);
    command(intern("showDream"), showDream);
    property(intern("neuronRows"), neuronRows);
    paint(intern("neuronPaint"), neuronPaint);
    paint(intern("drawingPaint"), drawingPaint);
    paint(intern("influencePaint"), influencePaint);

    property(intern("datasetNameText"), datasetNameText);
    property(intern("epochsText"), epochsText);
    property(intern("layersText"), layersText);
    property(intern("neuronsText"), neuronsText);
    property(intern("trainButtonText"), trainButtonText);
    property(intern("shiftAugment"), shiftAugment);
    property(intern("scaleAugment"), scaleAugment);
    property(intern("lastDatasetText"), lastDatasetText);
    property(intern("lastTrainText"), lastTrainText);
    property(intern("lastLossText"), lastLossText);
    property(intern("lastSavedText"), lastSavedText);
    property(intern("models"), models);
    command(intern("datasetPrev"), datasetPrev);
    command(intern("datasetNext"), datasetNext);
    command(intern("epochsUp"), epochsUp);
    command(intern("epochsDown"), epochsDown);
    command(intern("layersUp"), layersUp);
    command(intern("layersDown"), layersDown);
    command(intern("neuronsUp"), neuronsUp);
    command(intern("neuronsDown"), neuronsDown);
    command(intern("train"), train);
    command(intern("continueTraining"), continueTraining);
    command(intern("loadModel"), loadModel);
    command(intern("deleteModel"), deleteModel);
    command(intern("refreshModels"), refreshModels);

    property(intern("datasetTitleText"), datasetTitleText);
    property(intern("sampleInfoText"), sampleInfoText);
    property(intern("datasetCountsText"), datasetCountsText);
    property(intern("pageText"), pageText);
    property(intern("newDatasetName"), newDatasetName);
    property(intern("datasets"), datasets);
    property(intern("gridRows"), gridRows);
    command(intern("pagePrev"), pagePrev);
    command(intern("pageNext"), pageNext);
    command(intern("createDataset"), createDataset);
    command(intern("deleteDataset"), deleteDataset);
    command(intern("deleteSample"), deleteSample);

    property(intern("helpNav"), helpNav);
    // Codegen skips scroll-y, so the scroll offset of the topic text is registered here.
    property(intern("helpScroll"), helpScroll);
    for (std::size_t i = 0; i < kHelpTopicCount; ++i) {
        property(intern("helpTopic" + std::to_string(i)), helpTopic[i]);
        property(intern("helpShown" + std::to_string(i)), helpShown[i]);
    }
}

void RecognizerViewModel::show_tab(Tab tab) {
    tabRecognize = tab == Tab::Recognize;
    tabNeurons = tab == Tab::Neurons;
    tabTrain = tab == Tab::Train;
    tabDatasets = tab == Tab::Datasets;
    tabHelp = tab == Tab::Help;
}

}

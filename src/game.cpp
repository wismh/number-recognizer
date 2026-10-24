#include "game.h"

#include <asset_ids.h>

#include "loc/locale_prefs.h"

#include <engine/core/platform.h>
#include <engine/loc/catalog.h>
#include <engine/log.h>

#include <algorithm>
#include <filesystem>

namespace game {

namespace {

constexpr glm::vec2 kDesignSize{920.0f, 640.0f};

// The drawing area of recognizer.css (.draw-area), in design units: 16 padding, then the 44 tall tab bar, the 28 tall
// title and a 12 gap above it, so it starts at y = 16 + 44 + 28 + 12.
constexpr engine::render::Rect kDrawArea{16.0f, 100.0f, 280.0f, 280.0f};

}

Game::Game(const engine::EngineServices& services)
    : engine::GameBase(services.worlds)
    , jobs_(&services.jobs)
    , assets_(&services.assets) {}

engine::WindowDesc Game::primary_window() const {
    return engine::WindowDesc{
            .title = "MNIST Classifier", .size = {920, 640}, .style = {.resizable = false}};
}

std::optional<engine::AssetId> Game::window_icon() const {
    return assets::textures::app_icon;
}

void Game::on_start() {
    // MNIST IDX files ship with the game in <assets>/mnist (CMakeLists.txt copies data/mnist there). Saved models live
    // in <user data>/model, the user's own datasets in <user data>/datasets. Without a user data directory nothing can
    // be saved.
    std::filesystem::path data_directory;
    if (const auto* entry = assets_->catalog().find(assets::ui::recognizer)) {
        data_directory = entry->files_root / "mnist";
    }
    std::filesystem::path model_directory;
    std::filesystem::path dataset_directory;
    if (const auto directory = engine::user_data_directory("wismh", "number-recognizer")) {
        model_directory = *directory / "model";
        dataset_directory = *directory / "datasets";
        preferences_directory_ = *directory;
    } else {
        engine::log::warn("no user data directory: models cannot be saved");
    }

    load_locale();
    view_ = std::make_shared<RecognizerViewModel>();
    const auto& catalog = world().ctx<engine::loc::Catalog>();
    view_->localeIsEn = catalog.active() == "en";
    view_->localeIsUk = catalog.active() == "uk";
    view_->localeEn.bind_to<Game, &Game::use_english>(*this);
    view_->localeUk.bind_to<Game, &Game::use_ukrainian>(*this);
    recognizer_.emplace(*view_, *tr_, ModelStore{std::move(model_directory)},
            DatasetStore{std::move(dataset_directory)}, std::move(data_directory), *jobs_);
    help_.emplace(*view_, *tr_);

    engine::ui::UiCanvas canvas;
    canvas.document = assets::ui::recognizer;
    canvas.stylesheet = assets::css::recognizer;
    canvas.fit = engine::ui::UiFit::ScaleWithScreenSize;
    canvas.reference_size = kDesignSize;
    canvas.data_context = view_;
    screen_ = world().create();
    world().emplace<engine::ui::UiCanvas>(screen_, std::move(canvas));

    world().add_system(engine::ecs::Schedule::Frame, engine::ecs::Phase::Game,
            [this](engine::ecs::World& frame_world) { on_frame(frame_world); });
}

void Game::load_locale() {
    auto& catalog = world().ctx<engine::loc::Catalog>();
    catalog.add(*assets_->get<engine::loc::StringTable>(assets::locale::en), engine::loc::Role::Source);
    catalog.add(*assets_->get<engine::loc::StringTable>(assets::locale::uk));
    catalog.set_fallback("en");
    catalog.set_active(read_locale(preferences_directory_, "en"));
    tr_.emplace(catalog);
}

void Game::set_locale(std::string_view tag) {
    auto& catalog = world().ctx<engine::loc::Catalog>();
    if (catalog.active() == tag) {
        return;
    }
    catalog.set_active(std::string{tag});
    write_locale(preferences_directory_, tag);
    view_->localeIsEn = tag == "en";
    view_->localeIsUk = tag == "uk";
    recognizer_->apply_locale();
    help_->apply_locale();
}

void Game::on_frame(engine::ecs::World& world) {
    recognizer_->update();

    // The drawing area only exists on the Recognize tab; on the others the same spot shows something else.
    if (!view_->tabRecognize.get()) {
        drawing_ = false;
        last_point_.reset();
        return;
    }

    const engine::ui::UiPointer& pointer = engine::ui::pointer_for(world, engine::kPrimaryWindow);
    if (!pointer.down) {
        if (drawing_) {
            drawing_ = false;
            last_point_.reset();
            recognizer_->end_stroke();
        }
        return;
    }

    const auto design = to_design(world, pointer.position);
    if (!design) {
        return;
    }
    const bool inside = engine::ui::rect_contains(kDrawArea, design->x, design->y);
    if (!drawing_ && !inside) {
        return;  // the press started elsewhere (a button): dragging onto the canvas does not draw
    }
    drawing_ = true;
    if (!inside) {
        last_point_.reset();
        return;
    }

    // Fill the gap since the last frame so a fast stroke stays connected.
    const glm::vec2 point{(design->x - kDrawArea.x) / kDrawArea.w, (design->y - kDrawArea.y) / kDrawArea.h};
    const glm::vec2 from = last_point_.value_or(point);
    const int steps = std::max(1, static_cast<int>(glm::length(point - from) * 28.0f * 2.0f));
    for (int i = 1; i <= steps; ++i) {
        const glm::vec2 p = from + (point - from) * (static_cast<float>(i) / static_cast<float>(steps));
        recognizer_->draw_at(p.x, p.y);
    }
    last_point_ = point;
}

std::optional<glm::vec2> Game::to_design(engine::ecs::World& world, glm::vec2 window_position) const {
    const auto* canvas = world.try_get<engine::ui::UiCanvas>(screen_);
    if (canvas == nullptr || canvas->rect.w <= 0.0f) {
        return std::nullopt;
    }
    const engine::ui::UiCanvasSpace space = engine::ui::canvas_layout_space(canvas->rect, canvas->fit,
            canvas->reference_size);
    return (window_position - space.offset) / space.scale;
}

}

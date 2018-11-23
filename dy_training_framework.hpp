//
// Created by YAN Yuchen on 6/30/2018.
//

#ifndef DYNET_WRAPPER_DY_TRAINING_FRAMEWORK_HPP
#define DYNET_WRAPPER_DY_TRAINING_FRAMEWORK_HPP

#include <cmdlp.hpp>
#include <functional>
#include <dynet/dynet.h>
#include <dynet/io.h>
#include <json/json.h>
#include "dy.hpp"
#include <ctime>
#include <exception_util.hpp>
#include <word2vec.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

namespace tg {
  namespace dy {

    std::unordered_map<std::string, std::vector<float>> import_word2vec(const std::string &path) {
      std::unordered_map<std::string, std::vector<float> > embeddings;
      read_word2vec()(true,
                      util::id_func(),
                      util::curry(util::forced_map_lookup(), embeddings),
                      path);
      return embeddings;
    }

  /**
   * ensure that a directory exists, creating directories when necessary.
   * \param path
   */
    void ensure_dir(const std::string &path) {
      if (path.empty()) return;

      // recursively ensure parent directory
      std::size_t last_slash_pos = path.find_last_of('/');
      if (last_slash_pos != std::string::npos) ensure_dir(path.substr(0, last_slash_pos));

      // mkdir if doesn't exist
      struct stat st = {0};
      if (stat(path.c_str(), &st) == -1) {
        mkdir(path.c_str(), 0775);
      }
    }

    /**
     * get the filename of the latest file in a directory, according to the last modified time.
     * \param dir_path path to the directory
     * \param regex_filter a filter to match filename
     * \return the filename of the latest file, or empty of no matching files found
     */
    std::string get_latest_filename_in_directory(const std::string &dir_path, const std::string &regex_filter = "") {
      DIR *dp;
      struct dirent *dirp;
      if ((dp = opendir(dir_path.c_str())) == nullptr) {
        throw new std::runtime_error("Error( " + std::to_string(errno) + " ) opening " + dir_path);
      }

      time_t latest_modified_time = 0;
      std::string ret;
      while ((dirp = readdir(dp)) != nullptr) {
        std::string filename = std::string(dirp->d_name);

        // ignore the . and .. directory
        if (filename == "." || filename == "..") continue;

        // skip files that doesn't match the filter
        if (!regex_filter.empty() && !ECMAScript_string_utils::match(filename, regex_filter)) continue;

        // skip non-files
        struct stat file_stat;
        std::string file_path = dir_path + "/" + filename;
        if (stat(file_path.c_str(), &file_stat) == 0 && (file_stat.st_mode & S_IFREG)) {

          // store the latest filename
          auto modified_time_of_current_file = file_stat.st_mtim.tv_sec;
          if (modified_time_of_current_file > latest_modified_time) {
            latest_modified_time = modified_time_of_current_file;
            ret = filename;
          }
        }
      }
      closedir(dp);
      return ret;
    }

    class training_framework {
    public:

      struct training_options {
        std::string model_dir;
        unsigned num_epochs;
        unsigned num_workers;

        void init(cmdlp::parser &p) {
          p.add(cmdlp::value_option<std::string>(model_dir))
            ->name_value("directory path")
            ->desc("the directory where the model is stored")
            ->name("model_dir");
          p.add(cmdlp::value_option<unsigned>(num_epochs))
            ->fallback(10)
            ->name_value("natural number")
            ->desc("number of epoches to run")
            ->name("num_epochs");
          p.add(cmdlp::value_option<unsigned>(num_workers))
            ->fallback(4)
            ->name_value("natural number")
            ->desc("the number of parallel processes to spawn")
            ->name("num_workers");
        }
      };


      /**
       * train your model with small amount of data (which can fit into memory altogether)
       * \tparam model_T some sort of class that represents your model, must be able to export/import JSON structure
       * \tparam datum_T some sort of type that represents a training data item
       * \param o necessary command-line options
       * \param init initialize your model structure (will be called when there is no saved model to load)
       * \param compute_loss how to compute loss from your model, given a data item
       * \param data all training data
       */
      template<typename model_T, typename datum_T>
      static void train(const training_options &o,
                        std::function<model_T()> init,
                        std::function<dy::Tensor(const model_T &model, const datum_T &datum)> compute_loss,
                        const std::vector<datum_T> &data
      ) {
        using namespace std;
        model_T model = load_or_init_model(o.model_dir, init);
        for (unsigned epoch = 1; epoch <= o.num_epochs; epoch++) {
          std::cerr << "epoch " << epoch << std::endl;
          dy::fit<datum_T>(o.num_workers, data, [&](const datum_T &datum) { return compute_loss(model, datum); });
          save_model(o.model_dir, model);
        }
      }

      /**
       * train your model with large amount of data.
       * you need to (logically) split your data into several partitions,
       * then provide a list of functions in which each function can fetch a partition.
       * \tparam model_T some sort of class that represents your model, must be able to export/import JSON structure
       * \tparam datum_T some sort of type that represents a training data item
       * \param o necessary command-line options
       * \param init initialize your model structure  (will be called when there is no saved model to load)
       * \param compute_loss how to compute loss from your model, given a data item
       * \param data_fetchers a list of functions in which each function can fetch a partition
       */
      template<typename model_T, typename datum_T>
      static void train(const training_options &o,
                        std::function<model_T()> init,
                        std::function<dy::Tensor(const model_T &model, const datum_T &datum)> compute_loss,
                        const std::vector<std::function<std::vector<datum_T>()>> &data_fetchers
      ) {

        model_T model = load_or_init_model(o.model_dir, init);
        for (unsigned epoch = 1; epoch <= o.num_epochs; epoch++) {
          for (unsigned i = 0; i < data_fetchers.size(); i++) {
            std::cerr << "epoch " << epoch << ", partition " << (i + 1) << "/" << data_fetchers.size() << std::endl;
            dy::fit<datum_T>(o.num_workers, data_fetchers[i](), compute_loss);
            save_model(o.model_dir, model);
          }
        }
      }

    private:
      /**
       * load the latest model from model directory.
       * \tparam model_T some sort of class that represents your model
       * \return the loaded model
       */
      template<typename model_T>
      static model_T load_model(const std::string &model_dir) {
        model_T model;
        auto latest_model_filename = get_latest_filename_in_directory(model_dir, "\\.bin$");
        if (latest_model_filename.empty()) {
          throw std::runtime_error("load model: no model file found in path " + model_dir);
        }
        std::ifstream ifs(model_dir + "/" + latest_model_filename);
        cereal::BinaryInputArchive(ifs).operator()(model);
        return model;
      }

      template<typename model_T>
      static void save_model(const std::string &model_dir, const model_T &model) {
        std::time_t result = std::time(nullptr);
        std::ofstream ofs(model_dir + "/" + std::to_string(result) + ".bin");
        cereal::BinaryOutputArchive(ofs).operator()(model);
      }

      template<typename model_T>
      static model_T load_or_init_model(const std::string &model_dir, std::function<model_T()> init) {
        using namespace std;
        ensure_dir(model_dir);
        try {
          return load_model<model_T>(model_dir);
        }
        catch (std::runtime_error &ex) {
          cerr << "creating new model" << endl;
          return init();
        }
      }
    };
  }
}

#endif //DYNET_WRAPPER_DY_TRAINING_FRAMEWORK_HPP

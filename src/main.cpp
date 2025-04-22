#include <QTranslator>
#include <QLibraryInfo>
#include <qlocale.h>
#include <QApplication>
#include "main_window.h"

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  QLocale::setDefault(QLocale(QLocale::Russian, QLocale::Russia));
  QTranslator translator;
  if (translator.load("qtbase_ru",
                      QLibraryInfo::path(QLibraryInfo::TranslationsPath))) {
    app.installTranslator(&translator);
  }
  MainWindow window;
  window.show();
  return app.exec();
  return 0;
}
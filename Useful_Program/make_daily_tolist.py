from datetime import timedelta, date

from numpy.lib.arraysetops import isin


class MakeAutoDailiy(object):
    def __init__(self, start_dt, end_dt, day_add_subtitle={}):
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.__weekdays__ = [5, 6]
        self.check_start_dt()
        self.check_end_dt()
        self.day_add_subtitle = day_add_subtitle

    def intialize_list(
        self,
    ):
        self.one_week = []
        self.month_day_lists = []
        self.weekday_name_lists = []

    def check_start_dt(
        self,
    ):
        if self.start_dt.weekday() != 0:
            days_to_subtract = self.start_dt.weekday() - 0
            self.start_dt = self.start_dt - timedelta(days=days_to_subtract)

    def check_end_dt(
        self,
    ):
        if self.end_dt.weekday() != 4:
            days_to_subtract = self.end_dt.weekday() - 0
            self.end_dt = self.end_dt + timedelta(days=days_to_subtract)

    def daterange(self, date1, date2):
        for n in range(int((date2 - date1).days) + 1):
            yield date1 + timedelta(n)

    def run(
        self,
    ):
        seperate = []
        self.intialize_list()
        for dt in self.daterange(self.start_dt, self.end_dt):
            if dt.weekday() not in self.__weekdays__:
                weekday_name = dt.strftime("%A")
                month_day = dt.strftime("%m/%d")
                msg = f"    - {weekday_name}({month_day})"
                self.month_day_lists.append(month_day)
                self.weekday_name_lists.append(weekday_name)
                self.one_week.append(msg)
                if dt.strftime("%A").lower() in list(self.day_add_subtitle.keys()):
                    subtitles = self.day_add_subtitle[dt.strftime("%A").lower()]
                    for sub_title, comment in subtitles.items():
                        self.one_week.append(f"        - {sub_title}")
                        if comment is None:
                            continue
                        if isinstance(comment, str):
                            comment = [comment]
                        if isinstance(comment, list):
                            for com in comment:
                                self.one_week.append(f"            - {com}")
                if dt.strftime("%A").lower() == "friday":
                    self.one_week.insert(
                        0,
                        f"- {self.month_day_lists[0]}({self.weekday_name_lists[0]}) ~ {self.month_day_lists[-1]}({self.weekday_name_lists[-1]})",
                    )
                    seperate.append(self.one_week)
                    self.intialize_list()
                if dt == self.end_dt:
                    self.one_week.insert(
                        0,
                        f"- {self.month_day_lists[0]}({self.weekday_name_lists[0]}) ~ {self.month_day_lists[-1]}({self.weekday_name_lists[-1]})",
                    )
                    seperate.append(self.one_week)
                    self.intialize_list()
        print("복사해서 노션 혹은 드랍박스에 적용")
        return print("\n".join(["\n".join(sep) for sep in seperate]))


day_add_subtitle = {
    "friday": {"기타": "[주간](test2) 작성"},
    "tuesday": {"주간 회의": "[주간](test) 작성", "주간 회의2": ["test", "test2"], "test": None},
}
daily = MakeAutoDailiy(
    start_dt=date(2021, 6, 1),
    end_dt=date(2021, 7, 1),
    day_add_subtitle=day_add_subtitle,
)
daily.run()

import xlsxwriter


def xlsx_log_action(method_name, map_name, actions):
    workbook = xlsxwriter.Workbook('../log/' + method_name + '/' + map_name + '_action.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write_row('A1:A3', ['time', 'speed', 'steer'])

    col = 0
    for row, action in enumerate(actions):  # iterating through content list
        worksheet.write_row(row + 1, col, action)

    workbook.close()


def xlsx_log_observation(method_name, map_name, observations):
    workbook = xlsxwriter.Workbook('../log/' + method_name + '/' + map_name + '_observation.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write_row('A1:A5', ['time', 'x', 'y', 'theta', 'v_x'])

    col = 0
    for row, obs in enumerate(observations):  # iterating through content list
        worksheet.write_row(row + 1, col, obs)

    workbook.close()


def xlsx_log_error(method_name, map_name, errors):
    workbook = xlsxwriter.Workbook('../log/' + method_name + '/' + map_name + '_errors.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write_row('A1:A2', ['e_l', 'e_theta'])

    col = 0
    for row, err in enumerate(errors):  # iterating through content list
        worksheet.write_row(row + 1, col, err)

    workbook.close()
